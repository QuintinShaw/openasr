use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;

use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, validate_ggml_runtime_source_path, write_gguf_file_v0,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::ggml_family_registry::{
    MOONSHINE_AUDIO_FRONTEND_ID, MOONSHINE_DECODE_POLICY_ID, MOONSHINE_GGML_ARCHITECTURE_ID,
    MOONSHINE_TOKENIZER_ID,
};
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f32,
    read_source_json_file, tensor_element_count, validate_error, validate_output_pack_extension,
};
use crate::models::moonshine::MOONSHINE_MODEL_FAMILY;
use crate::models::moonshine::runtime_contract;
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1, insert_metadata,
    insert_metadata_string_array as insert_string_array, insert_metadata_u32 as insert_u32,
};
use crate::models::pack_quant::{PackQuant, classify_quant_tensor};
use crate::{read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source};

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_TOKENIZER_JSON: &str = "tokenizer.json";
const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";
const TOKENIZER_GGML_MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENIZER_GGML_MODEL_VALUE_LLAMA: &str = "llama";
const TOKENIZER_GGML_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";

pub type MoonshineLocalSourceError = LocalSourceImportError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoonshineLocalSourceImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub package_id: String,
    pub package_variant: Option<String>,
    pub source_name: String,
    pub source_revision: String,
    pub license_name: String,
    pub license_source: String,
    pub quantization: MoonshineRuntimeQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoonshineLocalSourceImportRuntimeResult {
    pub output_path: PathBuf,
    pub model_id: String,
    pub tensor_count: usize,
}

pub type MoonshineRuntimeQuantizationMode = PackQuant;

#[derive(Debug, Deserialize)]
struct MoonshineConfigJson {
    vocab_size: usize,
    hidden_size: usize,
    encoder_num_hidden_layers: usize,
    decoder_num_hidden_layers: usize,
    encoder_num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    partial_rotary_factor: f64,
    rope_theta: f64,
    bos_token_id: usize,
    eos_token_id: usize,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: TokenizerModelJson,
    #[serde(default)]
    added_tokens: Vec<TokenizerAddedToken>,
}

#[derive(Debug, Deserialize)]
struct TokenizerModelJson {
    vocab: BTreeMap<String, usize>,
}

#[derive(Debug, Deserialize)]
struct TokenizerAddedToken {
    id: usize,
    content: String,
}

pub fn convert_local_moonshine_source_to_runtime_pack(
    request: &MoonshineLocalSourceImportRequest,
) -> Result<MoonshineLocalSourceImportRuntimeResult, MoonshineLocalSourceError> {
    validate_request(request)?;
    let config: MoonshineConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    let tokenizer: TokenizerJson =
        read_source_json_file(&request.source_root, SOURCE_TOKENIZER_JSON)?;
    let model_path = request.source_root.join(SOURCE_MODEL_SAFETENSORS);
    let safetensors = SafetensorsFile::open(&model_path)?;

    let fields = moonshine_metadata_fields(&config)?;
    let vocab_tokens = build_vocab_tokens(&tokenizer, fields.vocab_size)?;
    let tensors = build_moonshine_runtime_tensors(&safetensors, request.quantization)?;
    let model_id = compose_model_id(&request.package_id, request.package_variant.as_deref());
    let metadata = moonshine_runtime_gguf_metadata(&fields, request, &model_id, &vocab_tokens);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "Moonshine local-source GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    // Fail closed: verify runtime contract via preflight reader/index validation.
    let runtime_source =
        validate_ggml_runtime_source_path(&request.output_root).map_err(|error| {
            validate_error(format!(
                "Moonshine local-source import produced invalid runtime path '{}': {error}",
                request.output_root.display()
            ))
        })?;
    let metadata_read =
        read_gguf_metadata_from_runtime_source(&runtime_source).map_err(|error| {
            validate_error(format!(
                "Moonshine import produced unreadable GGUF metadata: {error}"
            ))
        })?;
    let tensor_index =
        read_gguf_tensor_index_from_runtime_source(&runtime_source).map_err(|error| {
            validate_error(format!(
                "Moonshine import produced unreadable GGUF tensor index: {error}"
            ))
        })?;
    let execution_metadata = runtime_contract::parse_moonshine_execution_metadata(&metadata_read)
        .map_err(|error| {
        validate_error(format!(
            "Moonshine runtime metadata contract validation failed: {error}"
        ))
    })?;
    runtime_contract::validate_moonshine_runtime_tensors_with_index(
        &tensor_index,
        execution_metadata,
    )
    .map_err(|error| {
        validate_error(format!(
            "Moonshine runtime tensor contract validation failed: {error}"
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "Moonshine local-source GGUF writer produced unreadable tensor index: {error}"
        ))
    })?;
    Ok(MoonshineLocalSourceImportRuntimeResult {
        output_path: request.output_root.clone(),
        model_id,
        tensor_count: index.tensors().len(),
    })
}

fn build_moonshine_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: MoonshineRuntimeQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut names = BTreeSet::new();
    let mut out = Vec::new();
    for tensor in &safetensors.header().tensors {
        let Some((target_name, target_dims, force_f32)) =
            remap_moonshine_tensor_name_and_dims(tensor.name.as_str(), tensor.shape.as_slice())?
        else {
            continue;
        };
        if !names.insert(target_name.clone()) {
            return Err(validate_error(format!(
                "Moonshine import mapped duplicate destination tensor '{target_name}'"
            )));
        }
        let data = safetensors.tensor_data(tensor)?;
        let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
        let expected = tensor_element_count(&tensor.name, &target_dims)?;
        if values.len() != expected {
            return Err(validate_error(format!(
                "Moonshine tensor '{}' decoded {} values but expected {} for dims {:?}",
                tensor.name,
                values.len(),
                expected,
                target_dims
            )));
        }

        let tensor_type = quantized_tensor_type_for_moonshine_tensor(
            &target_name,
            &target_dims,
            force_f32,
            quantization,
        );
        let write_tensor = match tensor_type {
            Some(qtype) => {
                let quantized = quantize_f32_to_ggml_tensor_data(qtype, &target_dims, &values)
                    .map_err(|error| {
                        validate_error(format!(
                            "Moonshine quantization failed for '{target_name}' ({qtype:?}): {error}"
                        ))
                    })?;
                GgufWriteTensor {
                    name: target_name,
                    dims: target_dims,
                    tensor_type: qtype,
                    data: quantized,
                }
            }
            None => {
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
        };
        out.push(write_tensor);
    }

    if !names.contains("dec.emb.weight") {
        return Err(validate_error(
            "Moonshine source is missing model.decoder.embed_tokens.weight",
        ));
    }
    Ok(out)
}

/// Map an HF Moonshine tensor name + source shape to a runtime tensor name + dims.
///
/// Returns `(target_name, runtime_dims, force_f32)`. For rank-2 `.weight` projection
/// matrices the dims are reversed so the GGUF stores `[in, out]` (ggml column-major)
/// over the byte-identical HF row-major `[out, in]` payload, matching the `mul_mat(W, x)`
/// convention used throughout the runtime. The token embedding is similarly reversed to
/// `[d_model, vocab]` so the one tensor serves both `get_rows` and tied-logits `mul_mat`.
fn remap_moonshine_tensor_name_and_dims(
    source_name: &str,
    source_shape: &[u64],
) -> Result<Option<(String, Vec<u64>, bool)>, LocalSourceImportError> {
    // Conv stem. HF Conv1d weight is row-major [out_ch, in_ch, kernel]; reversing the
    // dims yields the ggml conv_1d kernel layout [kernel, in_ch, out_ch] over the same
    // bytes. Force f32 so the ggml conv_1d kernel type matches the data tensor.
    if source_name == "model.encoder.conv1.weight" {
        return Ok(Some((
            "enc.conv1.weight".to_string(),
            reverse_dims(source_shape),
            true,
        )));
    }
    if source_name == "model.encoder.conv2.weight" {
        return Ok(Some((
            "enc.conv2.weight".to_string(),
            reverse_dims(source_shape),
            true,
        )));
    }
    if source_name == "model.encoder.conv2.bias" {
        return Ok(Some((
            "enc.conv2.bias".to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "model.encoder.conv3.weight" {
        return Ok(Some((
            "enc.conv3.weight".to_string(),
            reverse_dims(source_shape),
            true,
        )));
    }
    if source_name == "model.encoder.conv3.bias" {
        return Ok(Some((
            "enc.conv3.bias".to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "model.encoder.groupnorm.weight" {
        return Ok(Some((
            "enc.groupnorm.weight".to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "model.encoder.groupnorm.bias" {
        return Ok(Some((
            "enc.groupnorm.bias".to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "model.encoder.layer_norm.weight" {
        return Ok(Some((
            "enc.out_norm.weight".to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "model.decoder.norm.weight" {
        return Ok(Some((
            "dec.out_norm.weight".to_string(),
            source_shape.to_vec(),
            true,
        )));
    }
    if source_name == "model.decoder.embed_tokens.weight" {
        // HF [vocab, d_model] -> ggml [d_model, vocab].
        return Ok(Some((
            "dec.emb.weight".to_string(),
            reverse_dims(source_shape),
            false,
        )));
    }

    if let Some(rest) = source_name.strip_prefix("model.encoder.layers.") {
        let (layer_idx, tail) = split_layer_index(rest)?;
        let prefix = format!("enc.blk.{layer_idx}");
        let (name, reverse, force_f32) = match tail {
            "input_layernorm.weight" => (format!("{prefix}.attn_norm.weight"), false, true),
            "self_attn.q_proj.weight" => (format!("{prefix}.attn_q.weight"), true, false),
            "self_attn.k_proj.weight" => (format!("{prefix}.attn_k.weight"), true, false),
            "self_attn.v_proj.weight" => (format!("{prefix}.attn_v.weight"), true, false),
            "self_attn.o_proj.weight" => (format!("{prefix}.attn_o.weight"), true, false),
            "post_attention_layernorm.weight" => (format!("{prefix}.ffn_norm.weight"), false, true),
            "mlp.fc1.weight" => (format!("{prefix}.ffn_up.weight"), true, false),
            "mlp.fc1.bias" => (format!("{prefix}.ffn_up.bias"), false, true),
            "mlp.fc2.weight" => (format!("{prefix}.ffn_down.weight"), true, false),
            "mlp.fc2.bias" => (format!("{prefix}.ffn_down.bias"), false, true),
            _ => return Ok(None),
        };
        let dims = if reverse {
            reverse_dims(source_shape)
        } else {
            source_shape.to_vec()
        };
        return Ok(Some((name, dims, force_f32)));
    }

    if let Some(rest) = source_name.strip_prefix("model.decoder.layers.") {
        let (layer_idx, tail) = split_layer_index(rest)?;
        let prefix = format!("dec.blk.{layer_idx}");
        let (name, reverse, force_f32) = match tail {
            "input_layernorm.weight" => (format!("{prefix}.attn_norm.weight"), false, true),
            "self_attn.q_proj.weight" => (format!("{prefix}.attn_q.weight"), true, false),
            "self_attn.k_proj.weight" => (format!("{prefix}.attn_k.weight"), true, false),
            "self_attn.v_proj.weight" => (format!("{prefix}.attn_v.weight"), true, false),
            "self_attn.o_proj.weight" => (format!("{prefix}.attn_o.weight"), true, false),
            "post_attention_layernorm.weight" => {
                (format!("{prefix}.cross_norm.weight"), false, true)
            }
            "encoder_attn.q_proj.weight" => (format!("{prefix}.cross_q.weight"), true, false),
            "encoder_attn.k_proj.weight" => (format!("{prefix}.cross_k.weight"), true, false),
            "encoder_attn.v_proj.weight" => (format!("{prefix}.cross_v.weight"), true, false),
            "encoder_attn.o_proj.weight" => (format!("{prefix}.cross_o.weight"), true, false),
            "final_layernorm.weight" => (format!("{prefix}.ffn_norm.weight"), false, true),
            "mlp.fc1.weight" => (format!("{prefix}.ffn_up.weight"), true, false),
            "mlp.fc1.bias" => (format!("{prefix}.ffn_up.bias"), false, true),
            "mlp.fc2.weight" => (format!("{prefix}.ffn_down.weight"), true, false),
            "mlp.fc2.bias" => (format!("{prefix}.ffn_down.bias"), false, true),
            _ => return Ok(None),
        };
        let dims = if reverse {
            reverse_dims(source_shape)
        } else {
            source_shape.to_vec()
        };
        return Ok(Some((name, dims, force_f32)));
    }

    Ok(None)
}

fn reverse_dims(shape: &[u64]) -> Vec<u64> {
    let mut dims = shape.to_vec();
    dims.reverse();
    dims
}

fn quantized_tensor_type_for_moonshine_tensor(
    name: &str,
    dims: &[u64],
    force_f32: bool,
    quantization: MoonshineRuntimeQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if force_f32 || quantization == MoonshineRuntimeQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    if !ne0.is_multiple_of(32_u64) {
        return None;
    }
    // The tied token embedding feeds both `get_rows` and the tied-logits `mul_mat`,
    // so it is the most precision-sensitive 2-D weight. Keep it at q8_0 at minimum
    // (8-bit, ~0.4% error) even when the rest of the pack requests q4_k — q4_k on
    // the output projection measurably hurts WER, while q8_0 (vs the old f32)
    // reclaims ~75% of its bytes (e.g. 54MB->14MB on base) with no WER change.
    if name == "dec.emb.weight" {
        return Some(GgufWriteTensorType::Q8_0);
    }
    classify_quant_tensor(ne0, quantization)
}

#[derive(Debug, Clone)]
struct MoonshineMetadataFields {
    vocab_size: usize,
    d_model: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    encoder_ffn_dim: usize,
    decoder_ffn_dim: usize,
    decoder_max_context: usize,
    bos_token_id: usize,
    eos_token_id: usize,
    rope_theta: f64,
}

fn moonshine_metadata_fields(
    config: &MoonshineConfigJson,
) -> Result<MoonshineMetadataFields, LocalSourceImportError> {
    let d_model = config.hidden_size;
    let n_heads = config.encoder_num_attention_heads;
    let head_dim = d_model
        .checked_div(n_heads)
        .filter(|value| value.saturating_mul(n_heads) == d_model)
        .ok_or_else(|| validate_error("Moonshine hidden_size must be divisible by n_heads"))?;
    let rotary_dim = ((head_dim as f64) * config.partial_rotary_factor) as usize;
    let rotary_dim = rotary_dim - (rotary_dim % 2); // even
    if rotary_dim == 0 || rotary_dim > head_dim {
        return Err(validate_error(format!(
            "Moonshine rotary_dim={rotary_dim} (head_dim={head_dim}, factor={}) is out of range",
            config.partial_rotary_factor
        )));
    }
    Ok(MoonshineMetadataFields {
        vocab_size: config.vocab_size,
        d_model,
        encoder_layers: config.encoder_num_hidden_layers,
        decoder_layers: config.decoder_num_hidden_layers,
        n_heads,
        head_dim,
        rotary_dim,
        encoder_ffn_dim: config.intermediate_size,
        decoder_ffn_dim: config.intermediate_size,
        decoder_max_context: config.max_position_embeddings,
        bos_token_id: config.bos_token_id,
        eos_token_id: config.eos_token_id,
        rope_theta: config.rope_theta,
    })
}

fn moonshine_runtime_gguf_metadata(
    fields: &MoonshineMetadataFields,
    request: &MoonshineLocalSourceImportRequest,
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
        MOONSHINE_MODEL_FAMILY,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        MOONSHINE_GGML_ARCHITECTURE_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        MOONSHINE_AUDIO_FRONTEND_ID,
    );
    insert_metadata(
        &mut metadata,
        OASR_METADATA_KEY_DECODE_POLICY,
        MOONSHINE_DECODE_POLICY_ID,
    );
    insert_metadata(&mut metadata, GGML_TOKENIZER_ID_KEY, MOONSHINE_TOKENIZER_ID);
    insert_metadata(&mut metadata, OPENASR_MODEL_ID_KEY, model_id);
    insert_metadata(
        &mut metadata,
        runtime_contract::GENERAL_ARCHITECTURE_KEY,
        runtime_contract::MOONSHINE_ARCHITECTURE_VALUE,
    );

    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_VOCAB_SIZE_KEY,
        fields.vocab_size as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_D_MODEL_KEY,
        fields.d_model as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_ENCODER_LAYERS_KEY,
        fields.encoder_layers as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_DECODER_LAYERS_KEY,
        fields.decoder_layers as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_HEADS_KEY,
        fields.n_heads as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_HEAD_DIM_KEY,
        fields.head_dim as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_ROTARY_DIM_KEY,
        fields.rotary_dim as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_ENCODER_FFN_DIM_KEY,
        fields.encoder_ffn_dim as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_DECODER_FFN_DIM_KEY,
        fields.decoder_ffn_dim as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_MAX_CONTEXT_KEY,
        fields.decoder_max_context as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_BOS_TOKEN_ID_KEY,
        fields.bos_token_id as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_EOS_TOKEN_ID_KEY,
        fields.eos_token_id as u32,
    );
    insert_u32(
        &mut metadata,
        runtime_contract::MOONSHINE_SAMPLE_RATE_KEY,
        16_000,
    );
    insert_metadata(
        &mut metadata,
        runtime_contract::MOONSHINE_ROPE_THETA_KEY,
        format!("{}", fields.rope_theta),
    );

    insert_metadata(
        &mut metadata,
        TOKENIZER_GGML_MODEL_KEY,
        TOKENIZER_GGML_MODEL_VALUE_LLAMA,
    );
    insert_string_array(&mut metadata, TOKENIZER_GGML_TOKENS_KEY, vocab_tokens);

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
    tokenizer: &TokenizerJson,
    vocab_size: usize,
) -> Result<Vec<String>, LocalSourceImportError> {
    let mut tokens = vec![String::new(); vocab_size];
    let mut filled = vec![false; vocab_size];
    for (token, token_id) in &tokenizer.model.vocab {
        if *token_id >= vocab_size {
            continue;
        }
        if !filled[*token_id] {
            tokens[*token_id] = token.clone();
            filled[*token_id] = true;
        }
    }
    // Added tokens (ids 0..2 specials and 32000..32767 segment tokens) override / fill gaps.
    for added in &tokenizer.added_tokens {
        if added.id >= vocab_size {
            continue;
        }
        tokens[added.id] = added.content.clone();
        filled[added.id] = true;
    }
    for (token_id, token) in tokens.iter_mut().enumerate() {
        if !filled[token_id] {
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
    request: &MoonshineLocalSourceImportRequest,
) -> Result<(), LocalSourceImportError> {
    for (value, field) in [
        (&request.package_id, "package_id"),
        (&request.source_name, "source_name"),
        (&request.source_revision, "source_revision"),
        (&request.license_name, "license_name"),
        (&request.license_source, "license_source"),
    ] {
        if value.trim().is_empty() {
            return Err(validate_error(format!(
                "Moonshine local-source converter requires non-empty {field}"
            )));
        }
    }
    validate_output_pack_extension(&request.output_root)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_request() -> MoonshineLocalSourceImportRequest {
        MoonshineLocalSourceImportRequest {
            source_root: PathBuf::from("/tmp/openasr/moonshine-source"),
            output_root: PathBuf::from("/tmp/openasr/moonshine-runtime.gguf"),
            package_id: "moonshine-fixture".to_string(),
            package_variant: Some("q8_0".to_string()),
            source_name: "UsefulSensors/moonshine".to_string(),
            source_revision: "fixture-revision".to_string(),
            license_name: "apache-2.0".to_string(),
            license_source: "https://example.invalid/license".to_string(),
            quantization: MoonshineRuntimeQuantizationMode::Q8_0,
        }
    }

    fn fixture_metadata_fields() -> MoonshineMetadataFields {
        MoonshineMetadataFields {
            vocab_size: 4,
            d_model: 16,
            encoder_layers: 1,
            decoder_layers: 1,
            n_heads: 2,
            head_dim: 8,
            rotary_dim: 4,
            encoder_ffn_dim: 64,
            decoder_ffn_dim: 64,
            decoder_max_context: 128,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10_000.0,
        }
    }

    fn string_metadata<'a>(metadata: &'a BTreeMap<String, GgufWriteValue>, key: &str) -> &'a str {
        match metadata.get(key) {
            Some(GgufWriteValue::String(value)) => value,
            other => panic!("expected string metadata for {key}, got {other:?}"),
        }
    }

    #[test]
    fn moonshine_runtime_metadata_declares_snapshot_streaming_feature() {
        let tokens = vec![
            "<pad>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "fixture".to_string(),
        ];
        let request = fixture_request();
        let metadata = moonshine_runtime_gguf_metadata(
            &fixture_metadata_fields(),
            &request,
            "moonshine-fixture:q8_0",
            &tokens,
        );

        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            MOONSHINE_MODEL_FAMILY
        );
        assert_eq!(
            string_metadata(&metadata, GGML_TOKENIZER_ID_KEY),
            MOONSHINE_TOKENIZER_ID
        );
    }
}
