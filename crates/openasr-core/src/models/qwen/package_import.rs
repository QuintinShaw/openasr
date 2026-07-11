use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::arch::hparams::QWEN3_ARCHITECTURE_VALUE;
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_metadata_from_runtime_source, read_gguf_tensor_index,
    read_gguf_tensor_index_from_runtime_source, validate_ggml_runtime_source_path,
    write_gguf_file_v0,
};
use crate::models::audio_frontend::mel::{FilterbankConfig, MelPointOrder, MelScale, filterbank};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::ggml_family_registry::{
    QWEN3_ASR_AUDIO_FRONTEND_ID, QWEN3_ASR_DECODE_POLICY_ID, QWEN3_ASR_GGML_ARCHITECTURE_ID,
    QWEN3_ASR_TOKENIZER_ID,
};
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f16_bits,
    decode_safetensors_payload_as_f32, encode_f16_bits_le, read_source_file_bytes,
    read_source_json_file, tensor_element_count, validate_error, validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
// Re-exported at `pub(super)` (not just imported) because `forced_aligner_import.rs`
// pulls these in via `use super::package_import::{insert_metadata, ...}`.
pub(super) use crate::models::oasr_metadata::{
    insert_metadata, insert_metadata_string_array, insert_metadata_u32,
};
use crate::models::pack_quant::{PackQuant, classify_quant_tensor};
use crate::models::runtime_tensor_contract_registry::validate_builtin_runtime_tensor_contract_for_architecture;
use crate::models::{qwen::QWEN3_ASR_MODEL_FAMILY, qwen::runtime_contract};

use super::tensor_names::{
    AUDIO_CONV_OUT_BIAS, AUDIO_CONV_OUT_WEIGHT, AUDIO_CONV1_BIAS, AUDIO_CONV1_WEIGHT,
    AUDIO_CONV2_BIAS, AUDIO_CONV2_WEIGHT, AUDIO_CONV3_BIAS, AUDIO_CONV3_WEIGHT, AUDIO_LN_POST_BIAS,
    AUDIO_LN_POST_WEIGHT, AUDIO_MEL_FILTERS, AUDIO_MEL_WINDOW, AUDIO_PROJ1_BIAS,
    AUDIO_PROJ1_WEIGHT, AUDIO_PROJ2_BIAS, AUDIO_PROJ2_WEIGHT, OUTPUT_NORM_WEIGHT, OUTPUT_WEIGHT,
    TOKEN_EMBD_WEIGHT, audio_layer_tensor_names, llm_layer_tensor_names,
};

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_VOCAB_JSON: &str = "vocab.json";
const SOURCE_MERGES_TXT: &str = "merges.txt";
const SOURCE_TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";
const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const TOKENIZER_GGML_MERGES_KEY: &str = "tokenizer.ggml.merges";
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";
const GENERAL_ARCHITECTURE_KEY: &str = "general.architecture";

pub type Qwen3AsrLocalSourceError = LocalSourceImportError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3AsrLocalSourceImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub package_id: String,
    pub package_variant: Option<String>,
    pub source_name: String,
    pub source_revision: String,
    pub license_name: String,
    pub license_source: String,
    pub quantization: Qwen3AsrRuntimeQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3AsrLocalSourceImportRuntimeResult {
    pub output_path: PathBuf,
    pub model_id: String,
    pub tensor_count: usize,
}

pub type Qwen3AsrRuntimeQuantizationMode = PackQuant;

#[derive(Debug, Deserialize)]
struct Qwen3AsrConfigJson {
    thinker_config: Qwen3AsrThinkerConfigJson,
}

#[derive(Debug, Deserialize)]
struct Qwen3AsrThinkerConfigJson {
    audio_config: Qwen3AsrAudioConfigJson,
    text_config: Qwen3AsrTextConfigJson,
    #[serde(default)]
    audio_token_id: Option<u32>,
    #[serde(default)]
    audio_start_token_id: Option<u32>,
    #[serde(default)]
    audio_end_token_id: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct Qwen3AsrAudioConfigJson {
    #[serde(default)]
    num_mel_bins: Option<usize>,
    #[serde(default)]
    encoder_layers: Option<usize>,
    #[serde(default)]
    d_model: Option<usize>,
    #[serde(default)]
    encoder_attention_heads: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Qwen3AsrTextConfigJson {
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfigJson {
    #[serde(default)]
    added_tokens_decoder: BTreeMap<String, AddedTokenEntry>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenEntry {
    content: String,
}

pub fn convert_local_qwen_source_to_runtime_pack(
    request: &Qwen3AsrLocalSourceImportRequest,
) -> Result<Qwen3AsrLocalSourceImportRuntimeResult, Qwen3AsrLocalSourceError> {
    validate_request(request)?;
    let config: Qwen3AsrConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    let mut tokens = load_vocab_tokens(&request.source_root)?;
    let merges = load_merges(&request.source_root)?;
    let metadata_fields = qwen_metadata_fields(&config, &tokens);
    if tokens.len() < metadata_fields.vocab_size {
        tokens.resize_with(metadata_fields.vocab_size, String::new);
    }
    patch_added_tokens(&request.source_root, &mut tokens)?;
    for (index, token) in tokens.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("<unused_{index}>");
        }
    }

    let safetensor_files = discover_safetensor_files(request)?;
    let mut tensors = build_qwen_runtime_tensors(
        &safetensor_files,
        request.quantization,
        metadata_fields.n_mels,
        metadata_fields.n_fft,
        metadata_fields.sample_rate_hz,
        metadata_fields.win_length,
    )?;

    let model_id = compose_model_id(&request.package_id, request.package_variant.as_deref());
    let metadata =
        qwen_runtime_gguf_metadata(request, &metadata_fields, &model_id, &tokens, &merges);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "Qwen local-source GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let runtime_source =
        validate_ggml_runtime_source_path(&request.output_root).map_err(|error| {
            validate_error(format!(
                "Qwen local-source import produced invalid runtime path '{}': {error}",
                request.output_root.display()
            ))
        })?;
    let metadata_read =
        read_gguf_metadata_from_runtime_source(&runtime_source).map_err(|error| {
            validate_error(format!(
                "Qwen import produced unreadable GGUF metadata: {error}"
            ))
        })?;
    let tensor_index =
        read_gguf_tensor_index_from_runtime_source(&runtime_source).map_err(|error| {
            validate_error(format!(
                "Qwen import produced unreadable GGUF tensor index: {error}"
            ))
        })?;
    validate_builtin_runtime_tensor_contract_for_architecture(
        QWEN3_ASR_GGML_ARCHITECTURE_ID,
        &metadata_read,
        &tensor_index,
    )
    .map_err(|error| validate_error(format!("Qwen runtime contract validation failed: {error}")))?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "Qwen local-source GGUF writer produced unreadable tensor index: {error}"
        ))
    })?;
    tensors.clear();
    Ok(Qwen3AsrLocalSourceImportRuntimeResult {
        output_path: request.output_root.clone(),
        model_id,
        tensor_count: index.tensors().len(),
    })
}

pub(super) fn build_qwen_runtime_tensors(
    safetensor_files: &[SafetensorsFile],
    quantization: Qwen3AsrRuntimeQuantizationMode,
    n_mels: usize,
    n_fft: usize,
    sample_rate_hz: u32,
    win_length: usize,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::new();
    // Synthesize frontend tensors as runtime contract inputs.
    let mel_filters = slaney_mel_filterbank(n_mels, n_fft, sample_rate_hz, 0.0, 8_000.0)?;
    out.push(f32_tensor(
        AUDIO_MEL_FILTERS,
        vec![n_mels as u64, (n_fft / 2 + 1) as u64],
        mel_filters,
    ));
    out.push(f32_tensor(
        AUDIO_MEL_WINDOW,
        vec![win_length as u64],
        hann_window(win_length),
    ));

    let mut seen = BTreeSet::new();
    seen.insert(AUDIO_MEL_FILTERS.to_string());
    seen.insert(AUDIO_MEL_WINDOW.to_string());
    for file in safetensor_files {
        for tensor in &file.header().tensors {
            let Some(mapped_name) = remap_qwen_tensor_name(&tensor.name)? else {
                continue;
            };
            if !seen.insert(mapped_name.clone()) {
                return Err(validate_error(format!(
                    "Qwen import mapped duplicate destination tensor '{mapped_name}'"
                )));
            }
            let target_dims = tensor.shape.clone();
            let data = file.tensor_data(tensor)?;
            let reverse_tensor_dims = should_reverse_qwen_tensor_dims(
                tensor.name.as_str(),
                target_dims.as_slice(),
                &mapped_name,
            );
            let effective_dims = if reverse_tensor_dims {
                target_dims.iter().copied().rev().collect()
            } else {
                target_dims.clone()
            };
            let force_f32 = is_qwen_f32_tensor(&mapped_name, target_dims.len());
            let qtype = quantized_tensor_type_for_qwen(
                &mapped_name,
                &effective_dims,
                force_f32,
                quantization,
            );
            let write_tensor = match qtype {
                Some(tensor_type) => {
                    let values =
                        decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                    let expected = tensor_element_count(&tensor.name, &effective_dims)?;
                    if values.len() != expected {
                        return Err(validate_error(format!(
                            "Qwen tensor '{}' decoded {} values but expected {} for dims {:?}",
                            tensor.name,
                            values.len(),
                            expected,
                            effective_dims
                        )));
                    }
                    let quantized = quantize_f32_to_ggml_tensor_data(
                        tensor_type,
                        &effective_dims,
                        &values,
                    )
                    .map_err(|error| {
                        validate_error(format!(
                            "Qwen quantization failed for tensor '{}' -> '{}' ({tensor_type:?}): {error}",
                            tensor.name, mapped_name
                        ))
                    })?;
                    GgufWriteTensor {
                        name: mapped_name,
                        dims: effective_dims,
                        tensor_type,
                        data: quantized,
                    }
                }
                None if force_f32 => {
                    let values =
                        decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                    let expected = tensor_element_count(&tensor.name, &effective_dims)?;
                    if values.len() != expected {
                        return Err(validate_error(format!(
                            "Qwen tensor '{}' decoded {} values but expected {} for dims {:?}",
                            tensor.name,
                            values.len(),
                            expected,
                            effective_dims
                        )));
                    }
                    f32_tensor(&mapped_name, effective_dims, values)
                }
                None => {
                    let bits =
                        decode_safetensors_payload_as_f16_bits(&tensor.name, &tensor.dtype, data)?;
                    GgufWriteTensor {
                        name: mapped_name,
                        dims: effective_dims,
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

fn remap_qwen_tensor_name(source_name: &str) -> Result<Option<String>, LocalSourceImportError> {
    let direct = match source_name {
        "thinker.audio_tower.conv2d1.weight" => Some(AUDIO_CONV1_WEIGHT),
        "thinker.audio_tower.conv2d1.bias" => Some(AUDIO_CONV1_BIAS),
        "thinker.audio_tower.conv2d2.weight" => Some(AUDIO_CONV2_WEIGHT),
        "thinker.audio_tower.conv2d2.bias" => Some(AUDIO_CONV2_BIAS),
        "thinker.audio_tower.conv2d3.weight" => Some(AUDIO_CONV3_WEIGHT),
        "thinker.audio_tower.conv2d3.bias" => Some(AUDIO_CONV3_BIAS),
        "thinker.audio_tower.conv_out.weight" => Some(AUDIO_CONV_OUT_WEIGHT),
        "thinker.audio_tower.conv_out.bias" => Some(AUDIO_CONV_OUT_BIAS),
        "thinker.audio_tower.ln_post.weight" => Some(AUDIO_LN_POST_WEIGHT),
        "thinker.audio_tower.ln_post.bias" => Some(AUDIO_LN_POST_BIAS),
        "thinker.audio_tower.proj1.weight" => Some(AUDIO_PROJ1_WEIGHT),
        "thinker.audio_tower.proj1.bias" => Some(AUDIO_PROJ1_BIAS),
        "thinker.audio_tower.proj2.weight" => Some(AUDIO_PROJ2_WEIGHT),
        "thinker.audio_tower.proj2.bias" => Some(AUDIO_PROJ2_BIAS),
        "thinker.model.embed_tokens.weight" => Some(TOKEN_EMBD_WEIGHT),
        "thinker.model.norm.weight" => Some(OUTPUT_NORM_WEIGHT),
        "thinker.lm_head.weight" => Some(OUTPUT_WEIGHT),
        _ => None,
    };
    if let Some(mapped) = direct {
        return Ok(Some(mapped.to_string()));
    }
    if let Some(rest) = source_name.strip_prefix("thinker.audio_tower.layers.") {
        let (layer_idx, tail) = split_layer_index(rest)?;
        let names = audio_layer_tensor_names(layer_idx);
        let mapped = match tail {
            "self_attn_layer_norm.weight" => names.attn_norm_weight,
            "self_attn_layer_norm.bias" => names.attn_norm_bias,
            "self_attn.q_proj.weight" => names.attn_q_weight,
            "self_attn.q_proj.bias" => names.attn_q_bias,
            "self_attn.k_proj.weight" => names.attn_k_weight,
            "self_attn.k_proj.bias" => names.attn_k_bias,
            "self_attn.v_proj.weight" => names.attn_v_weight,
            "self_attn.v_proj.bias" => names.attn_v_bias,
            "self_attn.out_proj.weight" => names.attn_out_weight,
            "self_attn.out_proj.bias" => names.attn_out_bias,
            "final_layer_norm.weight" => names.ffn_norm_weight,
            "final_layer_norm.bias" => names.ffn_norm_bias,
            "fc1.weight" => names.ffn_up_weight,
            "fc1.bias" => names.ffn_up_bias,
            "fc2.weight" => names.ffn_down_weight,
            "fc2.bias" => names.ffn_down_bias,
            _ => return Ok(None),
        };
        return Ok(Some(mapped));
    }
    if let Some(rest) = source_name.strip_prefix("thinker.model.layers.") {
        let (layer_idx, tail) = split_layer_index(rest)?;
        let names = llm_layer_tensor_names(layer_idx);
        let mapped = match tail {
            "input_layernorm.weight" => names.attn_norm_weight,
            "self_attn.q_proj.weight" => names.attn_q_weight,
            "self_attn.k_proj.weight" => names.attn_k_weight,
            "self_attn.v_proj.weight" => names.attn_v_weight,
            "self_attn.o_proj.weight" => names.attn_output_weight,
            "self_attn.q_norm.weight" => names.attn_q_norm_weight,
            "self_attn.k_norm.weight" => names.attn_k_norm_weight,
            "post_attention_layernorm.weight" => names.ffn_norm_weight,
            "mlp.gate_proj.weight" => names.ffn_gate_weight,
            "mlp.up_proj.weight" => names.ffn_up_weight,
            "mlp.down_proj.weight" => names.ffn_down_weight,
            _ => return Ok(None),
        };
        return Ok(Some(mapped));
    }
    Ok(None)
}

fn qwen_metadata_fields(config: &Qwen3AsrConfigJson, tokens: &[String]) -> Qwen3AsrMetadataFields {
    let audio = &config.thinker_config.audio_config;
    let text = &config.thinker_config.text_config;
    let llm_d_model = text.hidden_size.unwrap_or(1024);
    let llm_heads = text.num_attention_heads.unwrap_or(16);
    let llm_head_dim = text.head_dim.unwrap_or(llm_d_model / llm_heads.max(1));
    let vocab_size = text.vocab_size.unwrap_or(tokens.len()).max(tokens.len());
    Qwen3AsrMetadataFields {
        sample_rate_hz: 16_000,
        n_mels: audio.num_mel_bins.unwrap_or(128),
        n_fft: 400,
        win_length: 400,
        hop_length: 160,
        audio_layers: audio.encoder_layers.unwrap_or(18),
        audio_d_model: audio.d_model.unwrap_or(896),
        audio_heads: audio.encoder_attention_heads.unwrap_or(14),
        llm_layers: text.num_hidden_layers.unwrap_or(28),
        llm_d_model,
        llm_heads,
        llm_kv_heads: text.num_key_value_heads.unwrap_or(8),
        llm_head_dim,
        vocab_size,
        llm_max_positions: text.max_position_embeddings.unwrap_or(65_536),
        audio_start_token_id: config
            .thinker_config
            .audio_start_token_id
            .unwrap_or(151_669),
        audio_end_token_id: config.thinker_config.audio_end_token_id.unwrap_or(151_670),
        audio_pad_token_id: config.thinker_config.audio_token_id.unwrap_or(151_676),
        eos_token_id: 151_645,
        pad_token_id: 151_643,
    }
}

#[derive(Debug, Clone)]
struct Qwen3AsrMetadataFields {
    sample_rate_hz: u32,
    n_mels: usize,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    audio_layers: usize,
    audio_d_model: usize,
    audio_heads: usize,
    llm_layers: usize,
    llm_d_model: usize,
    llm_heads: usize,
    llm_kv_heads: usize,
    llm_head_dim: usize,
    vocab_size: usize,
    llm_max_positions: usize,
    audio_start_token_id: u32,
    audio_end_token_id: u32,
    audio_pad_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
}

fn qwen_runtime_gguf_metadata(
    request: &Qwen3AsrLocalSourceImportRequest,
    fields: &Qwen3AsrMetadataFields,
    model_id: &str,
    tokens: &[String],
    merges: &[String],
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
        QWEN3_ASR_MODEL_FAMILY,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        QWEN3_ASR_GGML_ARCHITECTURE_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        QWEN3_ASR_AUDIO_FRONTEND_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_DECODE_POLICY,
        QWEN3_ASR_DECODE_POLICY_ID,
    );
    insert_metadata(&mut metadata, GGML_TOKENIZER_ID_KEY, QWEN3_ASR_TOKENIZER_ID);
    insert_metadata(&mut metadata, OPENASR_MODEL_ID_KEY, model_id);
    insert_metadata(
        &mut metadata,
        GENERAL_ARCHITECTURE_KEY,
        QWEN3_ARCHITECTURE_VALUE,
    );

    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_SAMPLE_RATE_KEY,
        fields.sample_rate_hz,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_MELS_COUNT_KEY,
        fields.n_mels as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_N_FFT_KEY,
        fields.n_fft as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_WIN_LENGTH_KEY,
        fields.win_length as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_HOP_LENGTH_KEY,
        fields.hop_length as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_AUDIO_LAYERS_KEY,
        fields.audio_layers as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_AUDIO_D_MODEL_KEY,
        fields.audio_d_model as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_AUDIO_HEADS_KEY,
        fields.audio_heads as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_LLM_LAYERS_KEY,
        fields.llm_layers as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_LLM_D_MODEL_KEY,
        fields.llm_d_model as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_LLM_HEADS_KEY,
        fields.llm_heads as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_LLM_KV_HEADS_KEY,
        fields.llm_kv_heads as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_LLM_HEAD_DIM_KEY,
        fields.llm_head_dim as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_LLM_VOCAB_SIZE_KEY,
        fields.vocab_size as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_LLM_MAX_POSITIONS_KEY,
        fields.llm_max_positions as u32,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_AUDIO_START_TOKEN_ID_KEY,
        fields.audio_start_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_AUDIO_END_TOKEN_ID_KEY,
        fields.audio_end_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_AUDIO_PAD_TOKEN_ID_KEY,
        fields.audio_pad_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_EOS_TOKEN_ID_KEY,
        fields.eos_token_id,
    );
    insert_metadata_u32(
        &mut metadata,
        runtime_contract::QWEN3_PAD_TOKEN_ID_KEY,
        fields.pad_token_id,
    );

    insert_metadata(
        &mut metadata,
        TOKENIZER_GGML_MODEL_KEY,
        TOKENIZER_GGML_MODEL_VALUE_GPT2,
    );
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_TOKENS_KEY, tokens);
    insert_metadata_string_array(&mut metadata, TOKENIZER_GGML_MERGES_KEY, merges);

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

pub(super) fn patch_added_tokens(
    source_root: &Path,
    tokens: &mut [String],
) -> Result<(), LocalSourceImportError> {
    let path = source_root.join(SOURCE_TOKENIZER_CONFIG_JSON);
    if !path.exists() {
        return Ok(());
    }
    let cfg: TokenizerConfigJson =
        read_source_json_file(source_root, SOURCE_TOKENIZER_CONFIG_JSON)?;
    for (token_id_str, entry) in cfg.added_tokens_decoder {
        let token_id = token_id_str.parse::<usize>().map_err(|error| {
            validate_error(format!(
                "invalid tokenizer added token id '{}' in {}: {error}",
                token_id_str,
                path.display()
            ))
        })?;
        if token_id < tokens.len() {
            tokens[token_id] = entry.content;
        }
    }
    Ok(())
}

pub(super) fn load_vocab_tokens(source_root: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let vocab: BTreeMap<String, usize> = read_source_json_file(source_root, SOURCE_VOCAB_JSON)?;
    if vocab.is_empty() {
        return Err(validate_error("Qwen vocab.json cannot be empty"));
    }
    let mut pairs = vocab.into_iter().collect::<Vec<_>>();
    pairs.sort_by_key(|(_, token_id)| *token_id);
    let max_id = pairs
        .last()
        .map(|(_, token_id)| *token_id)
        .ok_or_else(|| validate_error("Qwen vocab.json cannot determine max token id"))?;
    let mut tokens = vec![String::new(); max_id + 1];
    for (token, token_id) in pairs {
        tokens[token_id] = token;
    }
    Ok(tokens)
}

pub(super) fn load_merges(source_root: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let path = source_root.join(SOURCE_MERGES_TXT);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = read_source_file_bytes(source_root, SOURCE_MERGES_TXT)?;
    let text = std::str::from_utf8(&bytes).map_err(|error| {
        validate_error(format!(
            "Qwen merges.txt is not valid UTF-8 ({}): {error}",
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

pub(super) fn discover_safetensor_files(
    request: &Qwen3AsrLocalSourceImportRequest,
) -> Result<Vec<SafetensorsFile>, LocalSourceImportError> {
    let mut paths = Vec::new();
    for entry in
        std::fs::read_dir(&request.source_root).map_err(|source| LocalSourceImportError::Read {
            path: request.source_root.clone(),
            source,
        })?
    {
        let entry = entry.map_err(|source| LocalSourceImportError::Read {
            path: request.source_root.clone(),
            source,
        })?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
        {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(validate_error(format!(
            "Qwen local-source converter could not find any *.safetensors in '{}'",
            request.source_root.display()
        )));
    }
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        files.push(SafetensorsFile::open(path)?);
    }
    Ok(files)
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

fn is_qwen_f32_tensor(name: &str, rank: usize) -> bool {
    rank <= 1 || name.ends_with(".bias") || name.contains("norm")
}

fn should_reverse_qwen_tensor_dims(
    source_name: &str,
    source_dims: &[u64],
    target_name: &str,
) -> bool {
    if source_dims.len() < 2 || !source_name.ends_with(".weight") {
        return false;
    }
    if source_name.starts_with("thinker.audio_tower.") {
        return true;
    }
    if source_name.starts_with("thinker.model.layers.") {
        return true;
    }
    target_name == TOKEN_EMBD_WEIGHT || target_name == OUTPUT_WEIGHT
}

fn quantized_tensor_type_for_qwen(
    name: &str,
    dims: &[u64],
    force_f32: bool,
    quantization: Qwen3AsrRuntimeQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if force_f32 || quantization == Qwen3AsrRuntimeQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    classify_quant_tensor(ne0, quantization)
}

fn f32_tensor(name: &str, dims: Vec<u64>, values: Vec<f32>) -> GgufWriteTensor {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    GgufWriteTensor {
        name: name.to_string(),
        dims,
        tensor_type: GgufWriteTensorType::F32,
        data: bytes,
    }
}

pub(super) fn compose_model_id(package_id: &str, package_variant: Option<&str>) -> String {
    match package_variant
        .map(str::trim)
        .filter(|variant| !variant.is_empty())
    {
        Some(variant) => format!("{}:{variant}", package_id.trim()),
        None => package_id.trim().to_string(),
    }
}

fn validate_request(
    request: &Qwen3AsrLocalSourceImportRequest,
) -> Result<(), LocalSourceImportError> {
    if request.package_id.trim().is_empty() {
        return Err(validate_error(
            "Qwen local-source converter requires non-empty package_id",
        ));
    }
    if request.source_name.trim().is_empty() {
        return Err(validate_error(
            "Qwen local-source converter requires non-empty source_name",
        ));
    }
    if request.source_revision.trim().is_empty() {
        return Err(validate_error(
            "Qwen local-source converter requires non-empty source_revision",
        ));
    }
    if request.license_name.trim().is_empty() {
        return Err(validate_error(
            "Qwen local-source converter requires non-empty license_name",
        ));
    }
    if request.license_source.trim().is_empty() {
        return Err(validate_error(
            "Qwen local-source converter requires non-empty license_source",
        ));
    }
    validate_output_pack_extension(&request.output_root)?;
    Ok(())
}

fn hann_window(length: usize) -> Vec<f32> {
    let scale = std::f32::consts::PI * 2.0 / length as f32;
    (0..length)
        .map(|index| 0.5 - 0.5 * (scale * index as f32).cos())
        .collect()
}

/// Slaney-scale filterbank, via the shared
/// [`crate::models::audio_frontend::mel`] `MelScale::Slaney` +
/// `MelPointOrder::RatioFirst` construction (the same hz<->mel warp, edge
/// placement, and ramp/area-normalization `whisper`'s frontend uses).
fn slaney_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> Result<Vec<f32>, LocalSourceImportError> {
    let fft_bins = n_fft / 2 + 1;
    let mel_major = filterbank(FilterbankConfig {
        scale: MelScale::Slaney,
        sample_rate_hz: sample_rate as f32,
        n_fft,
        n_mels,
        fmin,
        fmax,
        mel_point_order: MelPointOrder::RatioFirst,
    });
    // GGUF stores qwen audio.mel_filters with dims [n_mels, fft_bins], but
    // ggml's ne0/ne1 memory layout makes the contiguous payload freq-major.
    // Keep the runtime contract aligned with the working cstr reference:
    // transpose the shared module's mel-major `[n_mels, fft_bins]` matrix
    // into `[fft_bins, n_mels]` (bin-major, mel innermost).
    let mut filters = vec![0.0_f32; n_mels * fft_bins];
    for mel_idx in 0..n_mels {
        for bin_idx in 0..fft_bins {
            filters[bin_idx * n_mels + mel_idx] = mel_major[mel_idx * fft_bins + bin_idx];
        }
    }
    Ok(filters)
}

#[cfg(test)]
mod tests {
    use super::should_reverse_qwen_tensor_dims;

    #[test]
    fn qwen_tensor_dim_orientation_matches_runtime_contract() {
        assert!(should_reverse_qwen_tensor_dims(
            "thinker.model.layers.0.self_attn.q_proj.weight",
            &[2048, 1024],
            "blk.0.attn_q.weight",
        ));
        assert!(should_reverse_qwen_tensor_dims(
            "thinker.model.layers.0.mlp.down_proj.weight",
            &[1024, 3072],
            "blk.0.ffn_down.weight",
        ));
        assert!(should_reverse_qwen_tensor_dims(
            "thinker.model.embed_tokens.weight",
            &[151_936, 1024],
            "token_embd.weight",
        ));
        assert!(should_reverse_qwen_tensor_dims(
            "thinker.audio_tower.conv2d1.weight",
            &[480, 1, 3, 3],
            "audio.conv.1.weight",
        ));
        assert!(!should_reverse_qwen_tensor_dims(
            "thinker.model.layers.0.self_attn.q_proj.bias",
            &[2048],
            "blk.0.attn_q.bias",
        ));
        assert!(!should_reverse_qwen_tensor_dims(
            "thinker.model.layers.0.input_layernorm.weight",
            &[1024],
            "blk.0.attn_norm.weight",
        ));
    }

    #[test]
    fn qwen_mel_filterbank_uses_freq_major_payload_layout() {
        let filters = super::slaney_mel_filterbank(4, 8, 16_000, 0.0, 8_000.0).expect("filterbank");
        assert_eq!(filters.len(), 4 * (8 / 2 + 1));

        let rows = filters.chunks_exact(4).collect::<Vec<_>>();
        assert_eq!(rows.len(), 5);

        let first_bin_non_zero = rows
            .iter()
            .position(|row| row.iter().any(|value| value.abs() > f32::EPSILON))
            .expect("at least one non-zero row");
        assert!(first_bin_non_zero < rows.len());

        for row in rows {
            assert_eq!(row.len(), 4);
        }
    }

    /// Exact reimplementation of this importer's pre-shared-mel-module
    /// `slaney_mel_filterbank`/`hz_to_mel`/`mel_to_hz` (the version that
    /// shipped before it was switched to
    /// `crate::models::audio_frontend::mel`'s `MelScale::Slaney` +
    /// `MelPointOrder::RatioFirst` construction), kept only here to pin the
    /// baked `audio.mel_filters` tensor to a byte-identical value across the
    /// migration. Freq-major layout (`[fft_bins, n_mels]`), same as the live
    /// function.
    fn reference_pre_refactor_slaney_mel_filterbank(
        n_mels: usize,
        n_fft: usize,
        sample_rate: u32,
        fmin: f32,
        fmax: f32,
    ) -> Vec<f32> {
        fn hz_to_mel(hz: f32) -> f32 {
            let linear_scale = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / linear_scale;
            if hz < min_log_hz {
                hz / linear_scale
            } else {
                let log_step = 6.4_f32.ln() / 27.0;
                min_log_mel + (hz / min_log_hz).ln() / log_step
            }
        }
        fn mel_to_hz(mel: f32) -> f32 {
            let linear_scale = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / linear_scale;
            if mel < min_log_mel {
                mel * linear_scale
            } else {
                let log_step = 6.4_f32.ln() / 27.0;
                min_log_hz * (log_step * (mel - min_log_mel)).exp()
            }
        }
        let fft_bins = n_fft / 2 + 1;
        let fft_frequencies = (0..fft_bins)
            .map(|bin| bin as f32 * sample_rate as f32 / n_fft as f32)
            .collect::<Vec<_>>();
        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);
        let mel_points = (0..n_mels + 2)
            .map(|index| {
                let ratio = index as f32 / (n_mels + 1) as f32;
                mel_to_hz(mel_min + ratio * (mel_max - mel_min))
            })
            .collect::<Vec<_>>();
        let mut filters = vec![0.0_f32; n_mels * fft_bins];
        for mel_idx in 0..n_mels {
            let left = mel_points[mel_idx];
            let center = mel_points[mel_idx + 1];
            let right = mel_points[mel_idx + 2];
            let norm = 2.0 / (right - left).max(f32::EPSILON);
            for (bin_idx, hz) in fft_frequencies.iter().copied().enumerate() {
                let rising = (hz - left) / (center - left).max(f32::EPSILON);
                let falling = (right - hz) / (right - center).max(f32::EPSILON);
                filters[bin_idx * n_mels + mel_idx] = rising.min(falling).max(0.0) * norm;
            }
        }
        filters
    }

    #[test]
    fn mel_filterbank_is_byte_identical_to_pre_refactor_impl() {
        for n_mels in [80usize, 128usize] {
            let expected =
                reference_pre_refactor_slaney_mel_filterbank(n_mels, 400, 16_000, 0.0, 8_000.0);
            let actual = super::slaney_mel_filterbank(n_mels, 400, 16_000, 0.0, 8_000.0)
                .expect("filterbank");
            assert_eq!(expected, actual, "n_mels={n_mels}");
        }
    }
}
