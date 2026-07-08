//! Convert a local `nvidia/parakeet-tdt-*` HF source (safetensors + config.json
//! + tokenizer.json) into an OpenASR `.oasr` (GGUF-v0) runtime pack.
//!
//! Mirrors `parakeet_ctc::package_import` for the FastConformer encoder (the
//! SAME `enc.blk.{i}.*` / `enc.sub.*` conventions consumed by the shared
//! `nn::encoder::conformer_block`), with the v3 checkpoint's data differences:
//! no attention/conv/FFN biases (the loader synthesizes zeros; nothing is
//! fabricated into the pack), 128 mel bins, `scale_input: false`. The new TDT
//! tensors are the encoder joint projection (`enc.proj.*`), the 2-layer LSTM
//! prediction network (`dec.*`), the joint predictor projection
//! (`joint.pred.*`) and the fused `[vocab+blank | durations]` joint head
//! (`joint.out.*`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;

use crate::arch::{
    PARAKEET_TDT_AUDIO_FRONTEND_ID, PARAKEET_TDT_DECODE_POLICY_ID, PARAKEET_TDT_TOKENIZER_ID,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, write_gguf_file_v0,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f16_bits,
    decode_safetensors_payload_as_f32, encode_f16_bits_le, read_source_json_file, validate_error,
    validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::pack_quant::{PackQuant, classify_quant_tensor};

use super::runtime_contract::{
    PARAKEET_TDT_BLANK_TOKEN_ID_KEY, PARAKEET_TDT_CONV_KERNEL_KEY, PARAKEET_TDT_DURATIONS_KEY,
    PARAKEET_TDT_FFN_DIM_KEY, PARAKEET_TDT_HEAD_DIM_KEY, PARAKEET_TDT_HIDDEN_SIZE_KEY,
    PARAKEET_TDT_JOINT_HIDDEN_KEY, PARAKEET_TDT_MAX_SYMBOLS_PER_STEP_KEY,
    PARAKEET_TDT_N_DURATIONS_KEY, PARAKEET_TDT_N_HEADS_KEY, PARAKEET_TDT_N_LAYERS_KEY,
    PARAKEET_TDT_N_MELS_KEY, PARAKEET_TDT_PRED_HIDDEN_KEY, PARAKEET_TDT_PRED_LAYERS_KEY,
    PARAKEET_TDT_SCALE_INPUT_KEY, PARAKEET_TDT_SUBSAMPLING_CHANNELS_KEY,
    PARAKEET_TDT_SUBSAMPLING_FACTOR_KEY, PARAKEET_TDT_VOCAB_SIZE_KEY,
};
use super::{PARAKEET_TDT_GGML_ARCHITECTURE_ID, PARAKEET_TDT_MODEL_FAMILY};

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_TOKENIZER_JSON: &str = "tokenizer.json";
const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";

pub type ParakeetTdtQuantizationMode = PackQuant;

#[derive(Debug, Clone)]
pub struct ParakeetTdtImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: ParakeetTdtQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParakeetTdtImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub blank_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct ParakeetTdtConfigJson {
    encoder_config: ParakeetTdtEncoderConfigJson,
    vocab_size: usize,
    blank_token_id: u32,
    decoder_hidden_size: usize,
    num_decoder_layers: usize,
    durations: Vec<u32>,
    max_symbols_per_step: usize,
}

#[derive(Debug, Deserialize)]
struct ParakeetTdtEncoderConfigJson {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    conv_kernel_size: usize,
    num_mel_bins: usize,
    subsampling_factor: usize,
    subsampling_conv_channels: usize,
    scale_input: bool,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: TokenizerModelJson,
    #[serde(default)]
    added_tokens: Vec<TokenizerAddedToken>,
}

#[derive(Debug, Deserialize)]
struct TokenizerModelJson {
    vocab: BTreeMap<String, u32>,
}

#[derive(Debug, Deserialize)]
struct TokenizerAddedToken {
    id: u32,
    content: String,
}

pub fn convert_local_parakeet_tdt_source_to_runtime_pack(
    request: &ParakeetTdtImportRequest,
) -> Result<ParakeetTdtImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_root)?;
    let config: ParakeetTdtConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    let tokenizer: TokenizerJson =
        read_source_json_file(&request.source_root, SOURCE_TOKENIZER_JSON)?;
    let model_path = request.source_root.join(SOURCE_MODEL_SAFETENSORS);
    let safetensors = SafetensorsFile::open(&model_path)?;

    // The TDT decode loop uses the duration argmax INDEX as the frame skip;
    // that is only sound when the trained duration bins are exactly 0..n.
    // v3 ships [0,1,2,3,4]; fail closed on anything else.
    let contiguous = config
        .durations
        .iter()
        .enumerate()
        .all(|(index, &value)| value as usize == index);
    if config.durations.is_empty() || !contiguous {
        return Err(validate_error(format!(
            "parakeet-tdt durations {:?} must be the contiguous range 0..n",
            config.durations
        )));
    }
    if (config.blank_token_id as usize) + 1 != config.vocab_size {
        return Err(validate_error(format!(
            "parakeet-tdt blank_token_id {} must be the last vocab slot (vocab_size {})",
            config.blank_token_id, config.vocab_size
        )));
    }
    if config.num_decoder_layers != 2 {
        return Err(validate_error(format!(
            "parakeet-tdt importer supports the 2-layer LSTM predictor only, got {}",
            config.num_decoder_layers
        )));
    }

    let vocab_tokens = build_vocab_tokens(&tokenizer, config.vocab_size)?;
    let tensors = build_parakeet_tdt_runtime_tensors(&safetensors, request.quantization)?;
    let metadata = parakeet_tdt_runtime_gguf_metadata(&config, request, &vocab_tokens);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "parakeet-tdt GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "parakeet-tdt import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(ParakeetTdtImportResult {
        output_path: request.output_root.clone(),
        tensor_count: index.tensors().len(),
        blank_token_id: config.blank_token_id,
    })
}

/// Build the ordered `tokenizer.ggml.tokens` list (ids 0..=vocab_size-1) from
/// the BPE vocab + the added/special tokens (`<blank>` = the last id).
fn build_vocab_tokens(
    tokenizer: &TokenizerJson,
    vocab_size: usize,
) -> Result<Vec<String>, LocalSourceImportError> {
    let mut tokens = vec![None::<String>; vocab_size];
    for (token, &id) in &tokenizer.model.vocab {
        if (id as usize) < vocab_size {
            tokens[id as usize] = Some(token.clone());
        }
    }
    for added in &tokenizer.added_tokens {
        if (added.id as usize) < vocab_size {
            tokens[added.id as usize] = Some(added.content.clone());
        }
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| {
            token.ok_or_else(|| {
                validate_error(format!(
                    "parakeet-tdt tokenizer is missing token for id {id}"
                ))
            })
        })
        .collect()
}

fn build_parakeet_tdt_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: ParakeetTdtQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tensor in &safetensors.header().tensors {
        let Some(mapped) = remap_parakeet_tdt_tensor_name(tensor.name.as_str()) else {
            continue;
        };
        if !seen.insert(mapped.target_name.clone()) {
            return Err(validate_error(format!(
                "parakeet-tdt import mapped duplicate destination tensor '{}'",
                mapped.target_name
            )));
        }
        let target_dims = normalize_weight_dims(&mapped.target_name, tensor.shape.as_slice());
        let data = safetensors.tensor_data(tensor)?;
        let tensor_type = quantized_tensor_type_for_tensor(
            &mapped.target_name,
            &target_dims,
            mapped.storage,
            quantization,
        );
        let write_tensor = match tensor_type {
            Some(qtype) => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let quantized = quantize_f32_to_ggml_tensor_data(qtype, &target_dims, &values)
                    .map_err(|error| {
                        validate_error(format!(
                            "parakeet-tdt quantization failed for '{}' -> '{}' ({qtype:?}): {error}",
                            tensor.name, mapped.target_name
                        ))
                    })?;
                GgufWriteTensor {
                    name: mapped.target_name,
                    dims: target_dims,
                    tensor_type: qtype,
                    data: quantized,
                }
            }
            None if mapped.storage == TensorStorage::F32 => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                GgufWriteTensor {
                    name: mapped.target_name,
                    dims: target_dims,
                    tensor_type: GgufWriteTensorType::F32,
                    data: bytes,
                }
            }
            None => {
                let bits =
                    decode_safetensors_payload_as_f16_bits(&tensor.name, &tensor.dtype, data)?;
                GgufWriteTensor {
                    name: mapped.target_name,
                    dims: target_dims,
                    tensor_type: GgufWriteTensorType::F16,
                    data: encode_f16_bits_le(bits),
                }
            }
        };
        out.push(write_tensor);
    }
    Ok(out)
}

/// `enc.blk.{i}.{suffix}` — the conformer encoder-layer convention shared with
/// parakeet-ctc/cohere (consumed by `nn::encoder::conformer_block`).
fn enc_blk(layer: usize, suffix: &str) -> String {
    format!("enc.blk.{layer}.{suffix}")
}

/// Storage class for a mapped tensor when it is not quantized: `F32` for
/// norms/biases/conv/BN/subsampling (numerically sensitive or ggml-required
/// f32), `F16Quantizable` for the encoder 2-D linears (quantized when a quant
/// mode is selected), `F16` for the host-consumed predictor/joint weights
/// (kept f16 across quant modes — the greedy loop dequantizes to host f32
/// once at load; quantizing them would not change the resident footprint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorStorage {
    F32,
    F16,
    F16Quantizable,
}

struct MappedTensor {
    target_name: String,
    storage: TensorStorage,
}

fn mapped(target_name: String, storage: TensorStorage) -> Option<MappedTensor> {
    Some(MappedTensor {
        target_name,
        storage,
    })
}

/// Map a parakeet-tdt HF tensor name to its `.oasr` target name + storage
/// class. Returns `None` to drop a tensor (e.g. `num_batches_tracked`).
/// The v3 checkpoint has NO biases for the encoder projections/convs
/// (`attention_bias`/`convolution_bias`/FFN bias all absent), so unlike
/// parakeet-ctc there are no `.bias` arms for those here; the runtime loader
/// synthesizes zero biases for the shared conformer block.
fn remap_parakeet_tdt_tensor_name(source_name: &str) -> Option<MappedTensor> {
    // ----- TDT joint + prediction network -----
    match source_name {
        "encoder_projector.weight" => {
            return mapped("enc.proj.weight".to_string(), TensorStorage::F16Quantizable);
        }
        "encoder_projector.bias" => return mapped("enc.proj.bias".to_string(), TensorStorage::F32),
        "decoder.embedding.weight" => {
            return mapped("dec.embed.weight".to_string(), TensorStorage::F16);
        }
        "decoder.decoder_projector.weight" => {
            return mapped("joint.pred.weight".to_string(), TensorStorage::F16);
        }
        "decoder.decoder_projector.bias" => {
            return mapped("joint.pred.bias".to_string(), TensorStorage::F32);
        }
        "joint.head.weight" => return mapped("joint.out.weight".to_string(), TensorStorage::F16),
        "joint.head.bias" => return mapped("joint.out.bias".to_string(), TensorStorage::F32),
        _ => {}
    }
    if let Some(rest) = source_name.strip_prefix("decoder.lstm.") {
        // PyTorch LSTM naming: weight_ih_l{n} / weight_hh_l{n} / bias_ih_l{n} /
        // bias_hh_l{n}, gates packed [i|f|g|o] x hidden rows.
        let (kind, layer) = rest.rsplit_once("_l")?;
        let layer: usize = layer.parse().ok()?;
        let (target, storage) = match kind {
            "weight_ih" => ("w_ih", TensorStorage::F16),
            "weight_hh" => ("w_hh", TensorStorage::F16),
            "bias_ih" => ("b_ih", TensorStorage::F32),
            "bias_hh" => ("b_hh", TensorStorage::F32),
            _ => return None,
        };
        return mapped(format!("dec.lstm.{layer}.{target}"), storage);
    }
    // ----- FastConformer encoder (same conventions as parakeet-ctc) -----
    if let Some(rest) = source_name.strip_prefix("encoder.subsampling.") {
        return mapped(format!("enc.sub.{rest}"), TensorStorage::F32);
    }
    let rest = source_name.strip_prefix("encoder.layers.")?;
    let (layer, tail) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (suffix, storage) = match tail {
        "norm_feed_forward1.weight" => ("ff1.norm.weight", TensorStorage::F32),
        "norm_feed_forward1.bias" => ("ff1.norm.bias", TensorStorage::F32),
        "feed_forward1.linear1.weight" => ("ff1.up.weight", TensorStorage::F16Quantizable),
        "feed_forward1.linear2.weight" => ("ff1.down.weight", TensorStorage::F16Quantizable),
        "norm_self_att.weight" => ("attn.norm.weight", TensorStorage::F32),
        "norm_self_att.bias" => ("attn.norm.bias", TensorStorage::F32),
        "self_attn.q_proj.weight" => ("attn.q.weight", TensorStorage::F16Quantizable),
        "self_attn.k_proj.weight" => ("attn.k.weight", TensorStorage::F16Quantizable),
        "self_attn.v_proj.weight" => ("attn.v.weight", TensorStorage::F16Quantizable),
        "self_attn.o_proj.weight" => ("attn.out.weight", TensorStorage::F16Quantizable),
        "self_attn.relative_k_proj.weight" => ("attn.pos.weight", TensorStorage::F16Quantizable),
        "self_attn.bias_u" => ("attn.pos_bias_u", TensorStorage::F32),
        "self_attn.bias_v" => ("attn.pos_bias_v", TensorStorage::F32),
        "norm_conv.weight" => ("conv.norm.weight", TensorStorage::F32),
        "norm_conv.bias" => ("conv.norm.bias", TensorStorage::F32),
        "conv.pointwise_conv1.weight" => ("conv.pw1.weight", TensorStorage::F32),
        "conv.depthwise_conv.weight" => ("conv.dw.weight", TensorStorage::F32),
        "conv.norm.weight" => ("conv.bn.weight", TensorStorage::F32),
        "conv.norm.bias" => ("conv.bn.bias", TensorStorage::F32),
        "conv.norm.running_mean" => ("conv.bn.mean", TensorStorage::F32),
        "conv.norm.running_var" => ("conv.bn.var", TensorStorage::F32),
        "conv.pointwise_conv2.weight" => ("conv.pw2.weight", TensorStorage::F32),
        "norm_feed_forward2.weight" => ("ff2.norm.weight", TensorStorage::F32),
        "norm_feed_forward2.bias" => ("ff2.norm.bias", TensorStorage::F32),
        "feed_forward2.linear1.weight" => ("ff2.up.weight", TensorStorage::F16Quantizable),
        "feed_forward2.linear2.weight" => ("ff2.down.weight", TensorStorage::F16Quantizable),
        "norm_out.weight" => ("out.norm.weight", TensorStorage::F32),
        "norm_out.bias" => ("out.norm.bias", TensorStorage::F32),
        _ => return None,
    };
    mapped(enc_blk(layer, suffix), storage)
}

/// Reverse the dims of rank>=2 `.weight` tensors (HF `[out, in]` -> ggml
/// `[in, out]` for `mul_mat`; conv kernels `[OC, IC, kh, kw]` -> `[kw, kh, IC,
/// OC]`), plus the Transformer-XL `pos_bias_u/v` — exactly the parakeet-ctc /
/// cohere rule so every tensor feeding the shared conformer block keeps the
/// proven layout. The reversal is layout-free for the host-consumed
/// decoder/joint tensors (the flat buffer is unchanged; only the dim order
/// flips), so the rule is applied uniformly.
fn normalize_weight_dims(target_name: &str, source_shape: &[u64]) -> Vec<u64> {
    let reverse = target_name.ends_with(".weight")
        || target_name.contains("pos_bias")
        || target_name.contains("w_ih")
        || target_name.contains("w_hh");
    if reverse && source_shape.len() >= 2 {
        let mut dims = source_shape.to_vec();
        dims.reverse();
        dims
    } else {
        source_shape.to_vec()
    }
}

fn quantized_tensor_type_for_tensor(
    name: &str,
    dims: &[u64],
    storage: TensorStorage,
    quantization: ParakeetTdtQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if storage != TensorStorage::F16Quantizable || quantization == ParakeetTdtQuantizationMode::Fp16
    {
        return None;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    classify_quant_tensor(ne0, quantization)
}

fn parakeet_tdt_runtime_gguf_metadata(
    config: &ParakeetTdtConfigJson,
    request: &ParakeetTdtImportRequest,
    vocab_tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let enc = &config.encoder_config;
    let head_dim = enc.hidden_size / enc.num_attention_heads.max(1);
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str("general.architecture", PARAKEET_TDT_GGML_ARCHITECTURE_ID);
    put_str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put_str(OASR_METADATA_KEY_MODEL_FAMILY, PARAKEET_TDT_MODEL_FAMILY);
    put_str(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        PARAKEET_TDT_GGML_ARCHITECTURE_ID,
    );
    put_str(
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        PARAKEET_TDT_AUDIO_FRONTEND_ID,
    );
    put_str(
        OASR_METADATA_KEY_DECODE_POLICY,
        PARAKEET_TDT_DECODE_POLICY_ID,
    );
    put_str(GGML_TOKENIZER_ID_KEY, PARAKEET_TDT_TOKENIZER_ID);
    put_str("openasr.model.id", &request.model_id);

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32(PARAKEET_TDT_N_LAYERS_KEY, enc.num_hidden_layers as u32);
    put_u32(PARAKEET_TDT_HIDDEN_SIZE_KEY, enc.hidden_size as u32);
    put_u32(PARAKEET_TDT_N_HEADS_KEY, enc.num_attention_heads as u32);
    put_u32(PARAKEET_TDT_HEAD_DIM_KEY, head_dim as u32);
    put_u32(PARAKEET_TDT_FFN_DIM_KEY, enc.intermediate_size as u32);
    put_u32(PARAKEET_TDT_CONV_KERNEL_KEY, enc.conv_kernel_size as u32);
    put_u32(PARAKEET_TDT_N_MELS_KEY, enc.num_mel_bins as u32);
    put_u32(
        PARAKEET_TDT_SUBSAMPLING_FACTOR_KEY,
        enc.subsampling_factor as u32,
    );
    put_u32(
        PARAKEET_TDT_SUBSAMPLING_CHANNELS_KEY,
        enc.subsampling_conv_channels as u32,
    );
    put_u32(PARAKEET_TDT_SCALE_INPUT_KEY, u32::from(enc.scale_input));
    put_u32(PARAKEET_TDT_VOCAB_SIZE_KEY, config.vocab_size as u32);
    put_u32(PARAKEET_TDT_BLANK_TOKEN_ID_KEY, config.blank_token_id);
    put_u32(
        PARAKEET_TDT_PRED_HIDDEN_KEY,
        config.decoder_hidden_size as u32,
    );
    put_u32(
        PARAKEET_TDT_PRED_LAYERS_KEY,
        config.num_decoder_layers as u32,
    );
    // The joint hidden width equals the predictor projection output width
    // (both projections land in the same 640-wide joint space for v3).
    put_u32(
        PARAKEET_TDT_JOINT_HIDDEN_KEY,
        config.decoder_hidden_size as u32,
    );
    put_u32(PARAKEET_TDT_N_DURATIONS_KEY, config.durations.len() as u32);
    put_u32(
        PARAKEET_TDT_MAX_SYMBOLS_PER_STEP_KEY,
        config.max_symbols_per_step as u32,
    );
    metadata.insert(
        PARAKEET_TDT_DURATIONS_KEY.to_string(),
        GgufWriteValue::U32Array(config.durations.clone()),
    );
    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufWriteValue::StringArray(vocab_tokens.to_vec()),
    );
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fixture_config() -> ParakeetTdtConfigJson {
        ParakeetTdtConfigJson {
            encoder_config: ParakeetTdtEncoderConfigJson {
                hidden_size: 16,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                intermediate_size: 32,
                conv_kernel_size: 9,
                num_mel_bins: 128,
                subsampling_factor: 8,
                subsampling_conv_channels: 16,
                scale_input: false,
            },
            vocab_size: 5,
            blank_token_id: 4,
            decoder_hidden_size: 8,
            num_decoder_layers: 2,
            durations: vec![0, 1, 2, 3, 4],
            max_symbols_per_step: 10,
        }
    }

    #[test]
    fn remaps_tdt_predictor_and_joint_tensors() {
        let cases = [
            ("encoder_projector.weight", "enc.proj.weight"),
            ("decoder.embedding.weight", "dec.embed.weight"),
            ("decoder.lstm.weight_ih_l0", "dec.lstm.0.w_ih"),
            ("decoder.lstm.weight_hh_l1", "dec.lstm.1.w_hh"),
            ("decoder.lstm.bias_ih_l0", "dec.lstm.0.b_ih"),
            ("decoder.decoder_projector.weight", "joint.pred.weight"),
            ("joint.head.weight", "joint.out.weight"),
            ("joint.head.bias", "joint.out.bias"),
        ];
        for (source, expected) in cases {
            let mapped = remap_parakeet_tdt_tensor_name(source).expect(source);
            assert_eq!(mapped.target_name, expected);
        }
    }

    #[test]
    fn drops_bn_counter_and_unknown_tensors() {
        assert!(
            remap_parakeet_tdt_tensor_name("encoder.layers.0.conv.norm.num_batches_tracked")
                .is_none()
        );
        assert!(remap_parakeet_tdt_tensor_name("some.unknown.weight").is_none());
    }

    #[test]
    fn encoder_linears_are_quantizable_but_host_tensors_stay_f16() {
        let up = remap_parakeet_tdt_tensor_name("encoder.layers.3.feed_forward1.linear1.weight")
            .expect("ff1 up");
        assert_eq!(up.storage, TensorStorage::F16Quantizable);
        assert_eq!(
            quantized_tensor_type_for_tensor(
                &up.target_name,
                &[1024, 4096],
                up.storage,
                ParakeetTdtQuantizationMode::Q4_K,
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
        let head = remap_parakeet_tdt_tensor_name("joint.head.weight").expect("joint head");
        assert_eq!(head.storage, TensorStorage::F16);
        assert_eq!(
            quantized_tensor_type_for_tensor(
                &head.target_name,
                &[640, 8198],
                head.storage,
                ParakeetTdtQuantizationMode::Q4_K,
            ),
            None
        );
    }

    #[test]
    fn reverses_projection_and_lstm_dims() {
        assert_eq!(
            normalize_weight_dims("enc.proj.weight", &[640, 1024]),
            vec![1024, 640]
        );
        assert_eq!(
            normalize_weight_dims("dec.lstm.0.w_ih", &[2560, 640]),
            vec![640, 2560]
        );
        assert_eq!(normalize_weight_dims("joint.out.bias", &[8198]), vec![8198]);
    }

    #[test]
    fn metadata_declares_tdt_contract_keys() {
        let metadata = parakeet_tdt_runtime_gguf_metadata(
            &fixture_config(),
            &ParakeetTdtImportRequest {
                source_root: PathBuf::from("/tmp/src"),
                output_root: PathBuf::from("/tmp/out.oasr"),
                model_id: "parakeet-tdt-test".to_string(),
                quantization: ParakeetTdtQuantizationMode::Fp16,
            },
            &["a".to_string()],
        );
        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_FAMILY),
            Some(&GgufWriteValue::String("parakeet-tdt".to_string()))
        );
        assert_eq!(
            metadata.get(PARAKEET_TDT_BLANK_TOKEN_ID_KEY),
            Some(&GgufWriteValue::U32(4))
        );
        assert_eq!(
            metadata.get(PARAKEET_TDT_DURATIONS_KEY),
            Some(&GgufWriteValue::U32Array(vec![0, 1, 2, 3, 4]))
        );
        assert_eq!(
            metadata.get(PARAKEET_TDT_SCALE_INPUT_KEY),
            Some(&GgufWriteValue::U32(0))
        );
    }

    /// Producer (run with `--ignored`) that writes the host-local fp16 pack the
    /// encoder smoke + transcription gates consume. Not a gate; emits the file.
    #[test]
    #[ignore = "produces the host-local parakeet-tdt-0.6b-v3 .oasr pack"]
    fn produce_parakeet_tdt_06b_v3_fp16_pack() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/models/parakeet-tdt-0.6b-v3-source");
        if !source.join(SOURCE_MODEL_SAFETENSORS).exists() {
            eprintln!("skipping: parakeet-tdt-0.6b-v3 source not present");
            return;
        }
        let output = source.join("openasr/parakeet-tdt-0.6b-v3-fp16.oasr");
        std::fs::create_dir_all(output.parent().unwrap()).unwrap();
        let result = convert_local_parakeet_tdt_source_to_runtime_pack(&ParakeetTdtImportRequest {
            source_root: source,
            output_root: output.clone(),
            model_id: "parakeet-tdt-0.6b-v3".to_string(),
            quantization: ParakeetTdtQuantizationMode::Fp16,
        })
        .expect("parakeet-tdt fp16 pack");
        eprintln!(
            "wrote {} ({} tensors, blank {})",
            output.display(),
            result.tensor_count,
            result.blank_token_id
        );
    }
}
