//! Convert a local `facebook/wav2vec2-*` HF source (safetensors + config.json +
//! vocab.json) into an OpenASR `.oasr` (GGUF-v0) runtime pack.
//!
//! Mirrors `parakeet_ctc::package_import` (the same safetensors→GGUF path). The
//! genuinely-new bits: (1) the positional-conv weight-norm fold at import (one
//! effective `enc.posconv.weight` from `weight_g`/`weight_v`), (2) the
//! group-vs-layer feature-extractor norm handling (base uses group-norm on
//! layer 0 only), and (3) the char-CTC vocab built from `vocab.json` with the
//! pad token as the CTC blank.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;

use crate::arch::{
    WAV2VEC2_CTC_AUDIO_FRONTEND_ID, WAV2VEC2_CTC_DECODE_POLICY_ID, WAV2VEC2_CTC_TOKENIZER_ID,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, write_gguf_file_v0,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f32, encode_f16_bits_le,
    read_source_json_file, validate_error, validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1, OASR_METADATA_KEY_AUDIO_FRONTEND,
    OASR_METADATA_KEY_DECODE_POLICY, OASR_METADATA_KEY_FEATURE_STREAMING,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::nn::wav2vec2::fold_pos_conv_weight_norm;

use super::{WAV2VEC2_CTC_GGML_ARCHITECTURE_ID, WAV2VEC2_CTC_MODEL_FAMILY};

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_VOCAB_JSON: &str = "vocab.json";
const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";

const POS_CONV_WEIGHT_G: &str = "wav2vec2.encoder.pos_conv_embed.conv.weight_g";
const POS_CONV_WEIGHT_V: &str = "wav2vec2.encoder.pos_conv_embed.conv.weight_v";
const POS_CONV_BIAS: &str = "wav2vec2.encoder.pos_conv_embed.conv.bias";

/// The wav2vec2-family backbones publish tensors under different top-level
/// prefixes: `wav2vec2.` (wav2vec2/lv60), `hubert.` (HuBERT), and
/// `data2vec_audio.` (data2vec). The remapper canonicalizes everything to the
/// `wav2vec2.` prefix so one match table covers all variants.
const BACKBONE_PREFIXES: [&str; 3] = ["wav2vec2.", "hubert.", "data2vec_audio."];

/// Rewrite a source tensor name to the canonical `wav2vec2.`-prefixed form
/// (leaves non-backbone names like `lm_head.*` untouched).
fn canonicalize_backbone_prefix(name: &str) -> std::borrow::Cow<'_, str> {
    for prefix in BACKBONE_PREFIXES {
        if let Some(rest) = name.strip_prefix(prefix) {
            if prefix == "wav2vec2." {
                return std::borrow::Cow::Borrowed(name);
            }
            return std::borrow::Cow::Owned(format!("wav2vec2.{rest}"));
        }
    }
    std::borrow::Cow::Borrowed(name)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum Wav2Vec2CtcQuantizationMode {
    #[default]
    Fp16,
    Q8_0,
    Q4_K,
}

impl Wav2Vec2CtcQuantizationMode {
    fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q8_0 => "q8_0",
            Self::Q4_K => "q4_k",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Wav2Vec2CtcImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: Wav2Vec2CtcQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wav2Vec2CtcImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub blank_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct Wav2Vec2ConfigJson {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    pad_token_id: u32,
    num_conv_pos_embeddings: usize,
    num_conv_pos_embedding_groups: usize,
    /// `"group"` (base-960h) or `"layer"` (large variants). Defaults to "group".
    #[serde(default = "default_feat_extract_norm")]
    feat_extract_norm: String,
    /// Pre-norm stable-layer-norm encoder. Defaults to false (base-960h post-norm).
    #[serde(default)]
    do_stable_layer_norm: bool,
    /// Feature-extractor conv layers carry a bias. Defaults to false.
    #[serde(default)]
    conv_bias: bool,
    /// HF `model_type` — "wav2vec2"/"hubert" (single weight-norm pos-conv) or
    /// "data2vec-audio" (a stack of `num_conv_pos_embeddings` plain grouped convs).
    #[serde(default = "default_model_type")]
    model_type: String,
    /// data2vec pos-conv kernel size (the `num_conv_pos_embeddings` field is the
    /// LAYER COUNT for data2vec, not the kernel). Unused for wav2vec2/hubert.
    #[serde(default)]
    conv_pos_kernel_size: usize,
}

fn default_feat_extract_norm() -> String {
    "group".to_string()
}

fn default_model_type() -> String {
    "wav2vec2".to_string()
}

/// data2vec-audio stacks `num_conv_pos_embeddings` plain grouped convs as its
/// positional embedding (vs wav2vec2's single weight-norm conv).
fn config_is_data2vec(config: &Wav2Vec2ConfigJson) -> bool {
    config.model_type.contains("data2vec")
}

pub fn convert_local_wav2vec2_ctc_source_to_runtime_pack(
    request: &Wav2Vec2CtcImportRequest,
) -> Result<Wav2Vec2CtcImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_root)?;
    let config: Wav2Vec2ConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    let vocab: BTreeMap<String, u32> =
        read_source_json_file(&request.source_root, SOURCE_VOCAB_JSON)?;
    let model_path = request.source_root.join(SOURCE_MODEL_SAFETENSORS);
    let safetensors = SafetensorsFile::open(&model_path)?;

    let blank_token_id = config.pad_token_id;
    let vocab_tokens = build_vocab_tokens(&vocab, config.vocab_size)?;
    let tensors = build_wav2vec2_runtime_tensors(&safetensors, &config, request.quantization)?;
    let metadata = wav2vec2_runtime_gguf_metadata(&config, request, &vocab_tokens);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "wav2vec2-ctc GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "wav2vec2-ctc import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(Wav2Vec2CtcImportResult {
        output_path: request.output_root.clone(),
        tensor_count: index.tensors().len(),
        blank_token_id,
    })
}

/// Build the ordered `tokenizer.ggml.tokens` list (ids 0..=vocab_size-1) from the
/// char `vocab.json` (token -> id).
fn build_vocab_tokens(
    vocab: &BTreeMap<String, u32>,
    vocab_size: usize,
) -> Result<Vec<String>, LocalSourceImportError> {
    let mut tokens = vec![None::<String>; vocab_size];
    for (token, &id) in vocab {
        if (id as usize) < vocab_size {
            tokens[id as usize] = Some(token.clone());
        }
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| {
            token.ok_or_else(|| {
                validate_error(format!(
                    "wav2vec2-ctc tokenizer is missing token for id {id}"
                ))
            })
        })
        .collect()
}

fn build_wav2vec2_runtime_tensors(
    safetensors: &SafetensorsFile,
    config: &Wav2Vec2ConfigJson,
    quantization: Wav2Vec2CtcQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    let is_data2vec = config_is_data2vec(config);
    for tensor in &safetensors.header().tensors {
        let canonical = canonicalize_backbone_prefix(tensor.name.as_str());
        // The positional-conv weight_g/weight_v are folded into one tensor below;
        // skip them in the per-tensor loop.
        if canonical == POS_CONV_WEIGHT_G || canonical == POS_CONV_WEIGHT_V {
            continue;
        }
        // data2vec stacks its pos-conv as `...pos_conv_embed.layers.N.conv.*`;
        // emit those (weights + biases) separately in emit_data2vec_pos_conv.
        if is_data2vec && canonical.contains("encoder.pos_conv_embed.layers.") {
            continue;
        }
        let Some((target_name, force_f32)) = remap_wav2vec2_tensor_name(canonical.as_ref()) else {
            continue;
        };
        if !seen.insert(target_name.clone()) {
            return Err(validate_error(format!(
                "wav2vec2-ctc import mapped duplicate destination tensor '{target_name}'"
            )));
        }
        let target_dims = normalize_wav2vec2_weight_dims(&target_name, tensor.shape.as_slice());
        let data = safetensors.tensor_data(tensor)?;
        let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
        out.push(make_write_tensor(
            target_name,
            target_dims,
            values,
            force_f32,
            quantization,
        )?);
    }

    if is_data2vec {
        // data2vec: emit the stacked plain grouped convs as
        // enc.posconv.{i}.weight ([K,in/g,out] f16) + enc.posconv.{i}.bias (f32).
        for layer in emit_data2vec_pos_conv(safetensors, config)? {
            if !seen.insert(layer.name.clone()) {
                return Err(validate_error(format!(
                    "wav2vec2-ctc import produced a duplicate data2vec pos-conv tensor '{}'",
                    layer.name
                )));
            }
            out.push(layer);
        }
    } else {
        // wav2vec2/hubert: fold the positional-conv weight-norm into one effective
        // kernel and emit it in ggml `[K, in_per_group, out_channels]` layout
        // (reverse of PyTorch `[out, in_per_group, K]`), stored f16.
        let folded = fold_positional_conv(safetensors, config)?;
        if !seen.insert("enc.posconv.weight".to_string()) {
            return Err(validate_error(
                "wav2vec2-ctc import produced a duplicate enc.posconv.weight".to_string(),
            ));
        }
        out.push(folded);
    }

    Ok(out)
}

/// Emit data2vec's stacked positional-conv layers. Each
/// `data2vec_audio.encoder.pos_conv_embed.layers.{i}.conv.{weight,bias}` becomes
/// `enc.posconv.{i}.weight` (ggml `[K, in_per_group, out]` f16, the same
/// element-order-preserving dim reversal `fold_positional_conv` uses) and
/// `enc.posconv.{i}.bias` (f32). Plain conv — NO weight-norm fold.
fn emit_data2vec_pos_conv(
    safetensors: &SafetensorsFile,
    config: &Wav2Vec2ConfigJson,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let depth = config.num_conv_pos_embeddings;
    let find = |name: &str| {
        safetensors
            .header()
            .tensors
            .iter()
            .find(|t| canonicalize_backbone_prefix(t.name.as_str()) == name)
            .ok_or_else(|| validate_error(format!("data2vec pos-conv source missing '{name}'")))
    };
    let mut out = Vec::with_capacity(depth * 2);
    for i in 0..depth {
        let w_name = format!("wav2vec2.encoder.pos_conv_embed.layers.{i}.conv.weight");
        let b_name = format!("wav2vec2.encoder.pos_conv_embed.layers.{i}.conv.bias");
        let w_tensor = find(&w_name)?;
        if w_tensor.shape.len() != 3 {
            return Err(validate_error(format!(
                "data2vec pos-conv '{w_name}' rank {} != 3",
                w_tensor.shape.len()
            )));
        }
        // PyTorch conv weight [out, in_per_group, K]. Element order is identical
        // to ggml [K, in_per_group, out]; only the declared dims reverse.
        let out_channels = w_tensor.shape[0] as usize;
        let in_per_group = w_tensor.shape[1] as usize;
        let kernel = w_tensor.shape[2] as usize;
        let w_values = decode_safetensors_payload_as_f32(
            &w_tensor.name,
            &w_tensor.dtype,
            safetensors.tensor_data(w_tensor)?,
        )?;
        let bits: Vec<u16> = w_values.iter().copied().map(f32_to_f16_bits).collect();
        out.push(GgufWriteTensor {
            name: format!("enc.posconv.{i}.weight"),
            dims: vec![kernel as u64, in_per_group as u64, out_channels as u64],
            tensor_type: GgufWriteTensorType::F16,
            data: encode_f16_bits_le(bits),
        });
        let b_tensor = find(&b_name)?;
        let b_values = decode_safetensors_payload_as_f32(
            &b_tensor.name,
            &b_tensor.dtype,
            safetensors.tensor_data(b_tensor)?,
        )?;
        let mut b_bytes = Vec::with_capacity(b_values.len() * 4);
        for value in b_values {
            b_bytes.extend_from_slice(&value.to_le_bytes());
        }
        out.push(GgufWriteTensor {
            name: format!("enc.posconv.{i}.bias"),
            dims: vec![out_channels as u64],
            tensor_type: GgufWriteTensorType::F32,
            data: b_bytes,
        });
    }
    Ok(out)
}

/// Compute the weight-norm-folded positional conv kernel and emit it as a single
/// f16 `enc.posconv.weight` tensor with ggml dims `[K, in_per_group, out_channels]`.
fn fold_positional_conv(
    safetensors: &SafetensorsFile,
    config: &Wav2Vec2ConfigJson,
) -> Result<GgufWriteTensor, LocalSourceImportError> {
    let find = |name: &str| {
        safetensors
            .header()
            .tensors
            .iter()
            .find(|t| canonicalize_backbone_prefix(t.name.as_str()) == name)
            .ok_or_else(|| validate_error(format!("wav2vec2-ctc source missing '{name}'")))
    };
    let g_tensor = find(POS_CONV_WEIGHT_G)?;
    let v_tensor = find(POS_CONV_WEIGHT_V)?;
    let weight_g = decode_safetensors_payload_as_f32(
        &g_tensor.name,
        &g_tensor.dtype,
        safetensors.tensor_data(g_tensor)?,
    )?;
    let weight_v = decode_safetensors_payload_as_f32(
        &v_tensor.name,
        &v_tensor.dtype,
        safetensors.tensor_data(v_tensor)?,
    )?;
    // weight_v PyTorch shape [out_channels, in_per_group, kernel].
    if v_tensor.shape.len() != 3 {
        return Err(validate_error(format!(
            "wav2vec2-ctc pos-conv weight_v rank {} != 3",
            v_tensor.shape.len()
        )));
    }
    let out_channels = v_tensor.shape[0] as usize;
    let in_per_group = v_tensor.shape[1] as usize;
    let kernel = v_tensor.shape[2] as usize;
    if out_channels != config.hidden_size {
        return Err(validate_error(format!(
            "wav2vec2-ctc pos-conv out_channels {out_channels} != hidden_size {}",
            config.hidden_size
        )));
    }
    if kernel != config.num_conv_pos_embeddings {
        return Err(validate_error(format!(
            "wav2vec2-ctc pos-conv kernel {kernel} != num_conv_pos_embeddings {}",
            config.num_conv_pos_embeddings
        )));
    }
    // weight_g is [1,1,K]; flatten to [K].
    let g_flat: Vec<f32> = weight_g;
    let folded = fold_pos_conv_weight_norm(&weight_v, &g_flat, out_channels, in_per_group, kernel)
        .map_err(validate_error)?;
    // Re-layout from PyTorch C-order [out, in_per_group, K] (out outer, K inner)
    // to ggml conv_1d kernel layout [K, in_per_group, out] (K fastest, out outer).
    // ggml flat index = out*(in_per_group*K) + in*K + k  ==  PyTorch flat index,
    // so the BUFFER ORDER IS IDENTICAL; only the declared dims reverse. Store the
    // f16 bits in the same element order with reversed dims.
    let bits: Vec<u16> = folded.iter().copied().map(f32_to_f16_bits).collect();
    Ok(GgufWriteTensor {
        name: "enc.posconv.weight".to_string(),
        dims: vec![kernel as u64, in_per_group as u64, out_channels as u64],
        tensor_type: GgufWriteTensorType::F16,
        data: encode_f16_bits_le(bits),
    })
}

fn make_write_tensor(
    target_name: String,
    target_dims: Vec<u64>,
    values: Vec<f32>,
    force_f32: bool,
    quantization: Wav2Vec2CtcQuantizationMode,
) -> Result<GgufWriteTensor, LocalSourceImportError> {
    let tensor_type = quantized_tensor_type_for_wav2vec2_tensor(
        &target_name,
        &target_dims,
        force_f32,
        quantization,
    );
    Ok(match tensor_type {
        Some(qtype) => {
            let quantized = quantize_f32_to_ggml_tensor_data(qtype, &target_dims, &values)
                .map_err(|error| {
                    validate_error(format!(
                        "wav2vec2-ctc quantization failed for '{target_name}' ({qtype:?}): {error}"
                    ))
                })?;
            GgufWriteTensor {
                name: target_name,
                dims: target_dims,
                tensor_type: qtype,
                data: quantized,
            }
        }
        None if force_f32 => {
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
            let bits: Vec<u16> = values.iter().copied().map(f32_to_f16_bits).collect();
            GgufWriteTensor {
                name: target_name,
                dims: target_dims,
                tensor_type: GgufWriteTensorType::F16,
                data: encode_f16_bits_le(bits),
            }
        }
    })
}

/// `enc.blk.{i}.{suffix}` — the wav2vec2 post-norm encoder-layer convention.
fn enc_blk(layer: usize, suffix: &str) -> String {
    format!("enc.blk.{layer}.{suffix}")
}

/// Map a wav2vec2 HF tensor name to its `.oasr` target name + whether it must be
/// stored f32. Returns `None` to drop a tensor (e.g. `masked_spec_embed`).
fn remap_wav2vec2_tensor_name(source_name: &str) -> Option<(String, bool)> {
    if source_name == "lm_head.weight" {
        return Some(("ctc.head.weight".to_string(), false));
    }
    if source_name == "lm_head.bias" {
        return Some(("ctc.head.bias".to_string(), true));
    }
    if source_name == POS_CONV_BIAS {
        return Some(("enc.posconv.bias".to_string(), true));
    }
    if source_name == "wav2vec2.encoder.layer_norm.weight" {
        return Some(("enc.norm.weight".to_string(), true));
    }
    if source_name == "wav2vec2.encoder.layer_norm.bias" {
        return Some(("enc.norm.bias".to_string(), true));
    }
    // feature projection: layer_norm (over 512 channels) + Linear 512->768.
    if source_name == "wav2vec2.feature_projection.layer_norm.weight" {
        return Some(("enc.fp.norm.weight".to_string(), true));
    }
    if source_name == "wav2vec2.feature_projection.layer_norm.bias" {
        return Some(("enc.fp.norm.bias".to_string(), true));
    }
    if source_name == "wav2vec2.feature_projection.projection.weight" {
        return Some(("enc.fp.proj.weight".to_string(), false));
    }
    if source_name == "wav2vec2.feature_projection.projection.bias" {
        return Some(("enc.fp.proj.bias".to_string(), true));
    }
    // feature extractor conv layers.
    if let Some(rest) = source_name.strip_prefix("wav2vec2.feature_extractor.conv_layers.") {
        let (layer, tail) = rest.split_once('.')?;
        let layer: usize = layer.parse().ok()?;
        let suffix = match tail {
            "conv.weight" => "conv.weight",
            // conv bias (hubert/lv60 conv_bias=true).
            "conv.bias" => "conv.bias",
            // channel-norm gamma/beta: layer-0 group-norm (feat_extract_norm ==
            // "group") OR per-layer LayerNorm (feat_extract_norm == "layer").
            "layer_norm.weight" => "gn.weight",
            "layer_norm.bias" => "gn.bias",
            _ => return None,
        };
        let target = format!("enc.fe.{layer}.{suffix}");
        let force_f32 = wav2vec2_tensor_is_f32(&target);
        return Some((target, force_f32));
    }
    // encoder transformer layers.
    let rest = source_name.strip_prefix("wav2vec2.encoder.layers.")?;
    let (layer, tail) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let suffix = match tail {
        "attention.q_proj.weight" => "attn.q.weight",
        "attention.q_proj.bias" => "attn.q.bias",
        "attention.k_proj.weight" => "attn.k.weight",
        "attention.k_proj.bias" => "attn.k.bias",
        "attention.v_proj.weight" => "attn.v.weight",
        "attention.v_proj.bias" => "attn.v.bias",
        "attention.out_proj.weight" => "attn.out.weight",
        "attention.out_proj.bias" => "attn.out.bias",
        "layer_norm.weight" => "attn.norm.weight",
        "layer_norm.bias" => "attn.norm.bias",
        "feed_forward.intermediate_dense.weight" => "ffn.up.weight",
        "feed_forward.intermediate_dense.bias" => "ffn.up.bias",
        "feed_forward.output_dense.weight" => "ffn.down.weight",
        "feed_forward.output_dense.bias" => "ffn.down.bias",
        "final_layer_norm.weight" => "final.norm.weight",
        "final_layer_norm.bias" => "final.norm.bias",
        _ => return None,
    };
    let target = enc_blk(layer, suffix);
    let force_f32 = wav2vec2_tensor_is_f32(&target);
    Some((target, force_f32))
}

/// f32-required tensors: norms, biases, feature-extractor convs (f16-handled
/// elsewhere via the no-force path is fine, but conv kernels stay f16 not f32),
/// and the CTC head. Only the 2-D linear projections may be quantized.
fn wav2vec2_tensor_is_f32(target_name: &str) -> bool {
    target_name.ends_with(".bias")
        || target_name.contains(".norm.")
        || target_name.contains(".gn.")
        || target_name.starts_with("ctc.head")
}

/// Reverse the dims of 2-D+ projection/conv weights (HF `[out, in]` → ggml
/// `[in, out]` for `mul_mat`; HF conv `[OC, IC, K]` → ggml `[K, IC, OC]`).
fn normalize_wav2vec2_weight_dims(target_name: &str, source_shape: &[u64]) -> Vec<u64> {
    if should_reverse_wav2vec2_tensor_dims(target_name) && source_shape.len() >= 2 {
        let mut dims = source_shape.to_vec();
        dims.reverse();
        dims
    } else {
        source_shape.to_vec()
    }
}

fn should_reverse_wav2vec2_tensor_dims(target_name: &str) -> bool {
    // Every rank>=2 `.weight`: HF `[out,in]` linears -> ggml `[in,out]`, and HF
    // conv kernels `[OC, IC, K]` -> ggml `[K, IC, OC]`. 1-D biases/norms are
    // untouched (handled by the len < 2 guard in normalize).
    target_name.ends_with(".weight")
}

fn quantized_tensor_type_for_wav2vec2_tensor(
    name: &str,
    dims: &[u64],
    force_f32: bool,
    quantization: Wav2Vec2CtcQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if force_f32 || quantization == Wav2Vec2CtcQuantizationMode::Fp16 {
        return None;
    }
    // Only quantize 2-D linear `.weight` tensors (convs stay f16, handled by the
    // None branch). The feature-extractor convs are rank 3 so excluded here.
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    if !ne0.is_multiple_of(32_u64) {
        return None;
    }
    if quantization == Wav2Vec2CtcQuantizationMode::Q4_K && ne0.is_multiple_of(256_u64) {
        return Some(GgufWriteTensorType::Q4_K);
    }
    Some(GgufWriteTensorType::Q8_0)
}

fn wav2vec2_runtime_gguf_metadata(
    config: &Wav2Vec2ConfigJson,
    request: &Wav2Vec2CtcImportRequest,
    vocab_tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let head_dim = config.hidden_size / config.num_attention_heads.max(1);
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str("general.architecture", WAV2VEC2_CTC_GGML_ARCHITECTURE_ID);
    put_str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put_str(OASR_METADATA_KEY_MODEL_FAMILY, WAV2VEC2_CTC_MODEL_FAMILY);
    put_str(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        WAV2VEC2_CTC_GGML_ARCHITECTURE_ID,
    );
    put_str(
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        WAV2VEC2_CTC_AUDIO_FRONTEND_ID,
    );
    put_str(
        OASR_METADATA_KEY_DECODE_POLICY,
        WAV2VEC2_CTC_DECODE_POLICY_ID,
    );
    put_str(
        OASR_METADATA_KEY_FEATURE_STREAMING,
        OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1,
    );
    put_str(GGML_TOKENIZER_ID_KEY, WAV2VEC2_CTC_TOKENIZER_ID);
    put_str("openasr.model.id", &request.model_id);

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32("wav2vec2.n_layers", config.num_hidden_layers as u32);
    put_u32("wav2vec2.hidden_size", config.hidden_size as u32);
    put_u32("wav2vec2.n_heads", config.num_attention_heads as u32);
    put_u32("wav2vec2.head_dim", head_dim as u32);
    put_u32("wav2vec2.ffn_dim", config.intermediate_size as u32);
    put_u32("wav2vec2.vocab_size", config.vocab_size as u32);
    // For wav2vec2/hubert, `num_conv_pos_embeddings` IS the pos-conv kernel size
    // (single conv). For data2vec it is the LAYER COUNT, and the kernel lives in
    // `conv_pos_kernel_size`. Normalize so the runtime always reads the KERNEL
    // from `wav2vec2.num_conv_pos_embeddings` and the stack DEPTH from
    // `wav2vec2.pos_conv_depth` (absent/1 => single weight-norm conv).
    let is_data2vec = config_is_data2vec(config);
    let (pos_conv_kernel, pos_conv_depth) = if is_data2vec {
        (config.conv_pos_kernel_size, config.num_conv_pos_embeddings)
    } else {
        (config.num_conv_pos_embeddings, 1)
    };
    put_u32("wav2vec2.num_conv_pos_embeddings", pos_conv_kernel as u32);
    put_u32("wav2vec2.pos_conv_depth", pos_conv_depth as u32);
    put_u32(
        "wav2vec2.num_conv_pos_embedding_groups",
        config.num_conv_pos_embedding_groups as u32,
    );
    put_u32("ctc.blank_token_id", config.pad_token_id);
    // Resolved config flags that drive the runtime graph branches. data2vec-audio
    // ALWAYS uses a post-norm encoder (Data2VecAudioEncoderLayer norms AFTER attn/
    // FFN; the encoder applies layer_norm before the stack, no final norm) — its
    // config `do_stable_layer_norm` is an inherited field the model ignores, so
    // force post-norm here regardless of the config value.
    let do_stable_layer_norm = config.do_stable_layer_norm && !is_data2vec;
    put_u32(
        "wav2vec2.do_stable_layer_norm",
        u32::from(do_stable_layer_norm),
    );
    put_u32("wav2vec2.conv_bias", u32::from(config.conv_bias));

    metadata.insert(
        "wav2vec2.feat_extract_norm".to_string(),
        GgufWriteValue::String(config.feat_extract_norm.clone()),
    );
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufWriteValue::StringArray(vocab_tokens.to_vec()),
    );
    metadata
}

/// Round-to-nearest f32 -> f16 bit pattern (mirrors the cohere importer).
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa_with_hidden = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half_mantissa = (mantissa_with_hidden >> shift) as u16;
        let round_bit = 1_u32 << shift.saturating_sub(1);
        if (mantissa_with_hidden & round_bit) != 0 && (mantissa_with_hidden & (round_bit - 1)) != 0
        {
            half_mantissa += 1;
        }
        return sign | half_mantissa;
    }
    let mut half_mantissa = (mantissa >> 13) as u16;
    let round_bit = 1_u32 << 12;
    let mut half_exponent = half_exponent as u16;
    if (mantissa & round_bit) != 0 && (mantissa & (round_bit - 1)) != 0 {
        half_mantissa += 1;
        if half_mantissa == 0x400 {
            half_mantissa = 0;
            half_exponent += 1;
        }
    }
    sign | (half_exponent << 10) | half_mantissa
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixture_request() -> Wav2Vec2CtcImportRequest {
        Wav2Vec2CtcImportRequest {
            source_root: PathBuf::from("/tmp/wav2vec2-src"),
            output_root: PathBuf::from("/tmp/wav2vec2-out.oasr"),
            model_id: "wav2vec2-ctc-test".to_string(),
            quantization: Wav2Vec2CtcQuantizationMode::Q8_0,
        }
    }

    fn fixture_config() -> Wav2Vec2ConfigJson {
        Wav2Vec2ConfigJson {
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 32,
            vocab_size: 4,
            pad_token_id: 0,
            num_conv_pos_embeddings: 16,
            num_conv_pos_embedding_groups: 2,
            feat_extract_norm: "group".to_string(),
            do_stable_layer_norm: false,
            conv_bias: false,
            model_type: "wav2vec2".to_string(),
            conv_pos_kernel_size: 0,
        }
    }

    fn string_metadata(metadata: &BTreeMap<String, GgufWriteValue>, key: &str) -> Option<String> {
        match metadata.get(key) {
            Some(GgufWriteValue::String(value)) => Some(value.clone()),
            _ => None,
        }
    }

    #[test]
    fn wav2vec2_runtime_metadata_declares_snapshot_streaming_feature() {
        let metadata = wav2vec2_runtime_gguf_metadata(
            &fixture_config(),
            &fixture_request(),
            &["<pad>".to_string(), "a".to_string(), "b".to_string()],
        );

        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_FEATURE_STREAMING),
            Some(OASR_FEATURE_STREAMING_GGML_TRUE_STREAMING_V1.to_string())
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            Some(WAV2VEC2_CTC_MODEL_FAMILY.to_string())
        );
        assert_eq!(
            string_metadata(&metadata, GGML_TOKENIZER_ID_KEY),
            Some(WAV2VEC2_CTC_TOKENIZER_ID.to_string())
        );
    }

    fn source_root() -> Option<PathBuf> {
        [
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tmp/models/wav2vec2-base-960h-source"),
        ]
        .into_iter()
        .find(|p| p.join("model.safetensors").exists())
    }

    /// End-to-end round-trip on the real source (skipped when absent). Gates the
    /// importer: feature-extractor convs + group-norm, feature projection, folded
    /// pos-conv, 12 post-norm layers, encoder layer_norm, CTC head — all present,
    /// projection dims reversed to ggml `[in, out]`, blank id 0.
    ///
    /// Heavy: imports + quantizes the full safetensors. Ignored by default to keep
    /// the suite fast; run via `cargo nextest run --run-ignored all`.
    #[test]
    #[ignore = "heavy: imports+quantizes a real model; run via --run-ignored"]
    fn imports_wav2vec2_base_960h_round_trip_when_source_present() {
        let Some(source) = source_root() else {
            eprintln!("skipping: wav2vec2-base-960h source not present");
            return;
        };
        let output = std::env::temp_dir().join("oasr_wav2vec2_ctc_roundtrip.oasr");
        let _ = std::fs::remove_file(&output);
        let result = convert_local_wav2vec2_ctc_source_to_runtime_pack(&Wav2Vec2CtcImportRequest {
            source_root: source,
            output_root: output.clone(),
            model_id: "wav2vec2-base-960h-test".to_string(),
            quantization: Wav2Vec2CtcQuantizationMode::Fp16,
        })
        .expect("wav2vec2 import");
        assert_eq!(result.blank_token_id, 0);

        let index = read_gguf_tensor_index(&output).expect("read back index");
        let names: BTreeSet<&str> = index.tensors().iter().map(|t| t.name.as_str()).collect();
        let dims_of = |name: &str| -> Vec<u64> {
            index
                .tensors()
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("missing tensor {name}"))
                .dims
                .clone()
        };

        // feature extractor: 7 conv layers, group-norm gamma/beta on layer 0 only.
        for layer in 0..7 {
            assert!(
                names.contains(format!("enc.fe.{layer}.conv.weight").as_str()),
                "fe {layer}"
            );
        }
        assert!(names.contains("enc.fe.0.gn.weight"));
        assert!(names.contains("enc.fe.0.gn.bias"));
        assert!(!names.contains("enc.fe.1.gn.weight"));
        // feature projection + folded pos-conv + encoder layer_norm.
        assert!(names.contains("enc.fp.norm.weight"));
        assert!(names.contains("enc.fp.proj.weight"));
        assert!(names.contains("enc.posconv.weight"));
        assert!(names.contains("enc.posconv.bias"));
        assert!(names.contains("enc.norm.weight"));
        // 12 post-norm transformer layers.
        for layer in 0..12 {
            for suffix in [
                "attn.q.weight",
                "attn.k.weight",
                "attn.v.weight",
                "attn.out.weight",
                "attn.norm.weight",
                "ffn.up.weight",
                "ffn.down.weight",
                "final.norm.weight",
            ] {
                assert!(
                    names.contains(enc_blk(layer, suffix).as_str()),
                    "missing {layer}.{suffix}"
                );
            }
        }
        // ctc head present + correctly sized (reversed [in,out] = [768,32]).
        assert_eq!(dims_of("ctc.head.weight"), vec![768, 32]);
        assert_eq!(dims_of("ctc.head.bias"), vec![32]);
        // feature-extractor conv0 reversed [OC,IC,K]=[512,1,10] -> [10,1,512].
        assert_eq!(dims_of("enc.fe.0.conv.weight"), vec![10, 1, 512]);
        // folded pos-conv: ggml [K, in/g, out] = [128, 48, 768].
        assert_eq!(dims_of("enc.posconv.weight"), vec![128, 48, 768]);
        // ffn.up reversed [3072,768] -> [768,3072]; attn.q reversed [768,768].
        assert_eq!(dims_of(&enc_blk(0, "ffn.up.weight")), vec![768, 3072]);
        assert_eq!(dims_of(&enc_blk(0, "attn.q.weight")), vec![768, 768]);

        let _ = std::fs::remove_file(&output);
    }

    /// Producer (`--ignored`) for the canonical q4_k `.oasr` used by the WER gate.
    #[test]
    #[ignore = "produces the host-local wav2vec2-base-960h .oasr pack"]
    fn produce_wav2vec2_base_960h_q4k_pack() {
        let source = source_root().expect("wav2vec2-base-960h source");
        let output = source.join("openasr/wav2vec2-base-960h-q4k.oasr");
        std::fs::create_dir_all(output.parent().unwrap()).unwrap();
        let result = convert_local_wav2vec2_ctc_source_to_runtime_pack(&Wav2Vec2CtcImportRequest {
            source_root: source,
            output_root: output.clone(),
            model_id: "wav2vec2-base-960h".to_string(),
            quantization: Wav2Vec2CtcQuantizationMode::Q4_K,
        })
        .expect("wav2vec2 q4k pack");
        eprintln!(
            "wrote {} ({} tensors, blank {})",
            output.display(),
            result.tensor_count,
            result.blank_token_id
        );
    }

    /// The backbone-prefix canonicalizer maps `hubert.`/`data2vec_audio.` to the
    /// `wav2vec2.` prefix so one remap table covers every sibling; non-backbone
    /// names (`lm_head.*`) pass through unchanged.
    #[test]
    fn canonicalizes_sibling_backbone_prefixes() {
        assert_eq!(
            canonicalize_backbone_prefix("hubert.encoder.layers.3.attention.q_proj.weight"),
            "wav2vec2.encoder.layers.3.attention.q_proj.weight"
        );
        assert_eq!(
            canonicalize_backbone_prefix(
                "data2vec_audio.feature_extractor.conv_layers.0.conv.weight"
            ),
            "wav2vec2.feature_extractor.conv_layers.0.conv.weight"
        );
        assert_eq!(
            canonicalize_backbone_prefix("wav2vec2.encoder.pos_conv_embed.conv.weight_g"),
            "wav2vec2.encoder.pos_conv_embed.conv.weight_g"
        );
        assert_eq!(
            canonicalize_backbone_prefix("lm_head.weight"),
            "lm_head.weight"
        );
    }

    fn hubert_source_root() -> Option<PathBuf> {
        [Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/models/hubert-large-ls960-ft-source")]
        .into_iter()
        .find(|p| p.join("model.safetensors").exists())
    }

    /// Large "layer"-variant round-trip (HuBERT): per-conv-layer LayerNorm
    /// gamma/beta on ALL 7 layers, conv bias on every layer, the `hubert.`
    /// backbone prefix canonicalized, and the resolved config flags written.
    /// Skipped when the source is absent.
    ///
    /// Heavy: imports + quantizes hubert-large. Ignored by default; run via
    /// `cargo nextest run --run-ignored all`.
    #[test]
    #[ignore = "heavy: imports+quantizes a real model; run via --run-ignored"]
    fn imports_hubert_large_layer_variant_when_source_present() {
        let Some(source) = hubert_source_root() else {
            eprintln!("skipping: hubert-large source not present");
            return;
        };
        let output = std::env::temp_dir().join("oasr_hubert_large_roundtrip.oasr");
        let _ = std::fs::remove_file(&output);
        convert_local_wav2vec2_ctc_source_to_runtime_pack(&Wav2Vec2CtcImportRequest {
            source_root: source,
            output_root: output.clone(),
            model_id: "hubert-large-test".to_string(),
            quantization: Wav2Vec2CtcQuantizationMode::Fp16,
        })
        .expect("hubert import");

        let index = read_gguf_tensor_index(&output).expect("read back index");
        let names: BTreeSet<&str> = index.tensors().iter().map(|t| t.name.as_str()).collect();
        // "layer" variant: per-conv-layer LayerNorm + conv bias on EVERY layer.
        for layer in 0..7 {
            assert!(
                names.contains(format!("enc.fe.{layer}.gn.weight").as_str()),
                "gn {layer}"
            );
            assert!(
                names.contains(format!("enc.fe.{layer}.conv.bias").as_str()),
                "cb {layer}"
            );
        }
        // standard grouped pos-conv folded, 24 transformer layers.
        assert!(names.contains("enc.posconv.weight"));
        for suffix in ["attn.q.weight", "ffn.up.weight", "final.norm.weight"] {
            assert!(
                names.contains(enc_blk(23, suffix).as_str()),
                "missing 23.{suffix}"
            );
        }
        let _ = std::fs::remove_file(&output);
    }
}
