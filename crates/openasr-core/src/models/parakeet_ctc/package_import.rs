//! Convert a local `nvidia/parakeet-ctc-*` HF source (safetensors + config.json
//! + tokenizer.json) into an OpenASR `.oasr` (GGUF-v0) runtime pack.
//!
//! Mirrors `cohere::package_import` (the same safetensors→GGUF path) — the
//! conformer encoder-layer tensors map onto the SAME `enc.blk.{i}.*` convention
//! the shared `nn::encoder::conformer_block` consumes (so the per-layer math is
//! the already-proven cohere block). The genuinely-new tensors are the
//! dw-striding subsampling prelude (`enc.sub.*`), the CTC head (`ctc.head.*`),
//! and the BatchNorm tensors folded into the depthwise at runtime load.

// The importer is exercised by the round-trip test + the #[ignore] pack producer
// until S4 wires a `model-pack import-parakeet-ctc` CLI command + the executor.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;

use crate::arch::{
    PARAKEET_CTC_AUDIO_FRONTEND_ID, PARAKEET_CTC_DECODE_POLICY_ID, PARAKEET_CTC_TOKENIZER_ID,
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

use super::{PARAKEET_CTC_GGML_ARCHITECTURE_ID, PARAKEET_CTC_MODEL_FAMILY};

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_TOKENIZER_JSON: &str = "tokenizer.json";
const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum ParakeetCtcQuantizationMode {
    #[default]
    Fp16,
    Q8_0,
    Q4_K,
}

impl ParakeetCtcQuantizationMode {
    fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q8_0 => "q8_0",
            Self::Q4_K => "q4_k",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParakeetCtcImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: ParakeetCtcQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParakeetCtcImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub blank_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct ParakeetConfigJson {
    encoder_config: ParakeetEncoderConfigJson,
    vocab_size: usize,
    pad_token_id: u32,
}

#[derive(Debug, Deserialize)]
struct ParakeetEncoderConfigJson {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    conv_kernel_size: usize,
    num_mel_bins: usize,
    subsampling_factor: usize,
    subsampling_conv_channels: usize,
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

pub fn convert_local_parakeet_ctc_source_to_runtime_pack(
    request: &ParakeetCtcImportRequest,
) -> Result<ParakeetCtcImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_root)?;
    let config: ParakeetConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    let tokenizer: TokenizerJson =
        read_source_json_file(&request.source_root, SOURCE_TOKENIZER_JSON)?;
    let model_path = request.source_root.join(SOURCE_MODEL_SAFETENSORS);
    let safetensors = SafetensorsFile::open(&model_path)?;

    let blank_token_id = config.pad_token_id;
    let vocab_tokens = build_vocab_tokens(&tokenizer, config.vocab_size)?;
    let tensors = build_parakeet_runtime_tensors(&safetensors, request.quantization)?;
    let metadata = parakeet_runtime_gguf_metadata(&config, request, &vocab_tokens);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "parakeet-ctc GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "parakeet-ctc import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(ParakeetCtcImportResult {
        output_path: request.output_root.clone(),
        tensor_count: index.tensors().len(),
        blank_token_id,
    })
}

/// Build the ordered `tokenizer.ggml.tokens` list (ids 0..=vocab_size-1), filling
/// from the BPE vocab + the added/special tokens (e.g. `<pad>` = the blank id).
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
                    "parakeet-ctc tokenizer is missing token for id {id}"
                ))
            })
        })
        .collect()
}

fn build_parakeet_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: ParakeetCtcQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tensor in &safetensors.header().tensors {
        let Some((target_name, force_f32)) = remap_parakeet_tensor_name(tensor.name.as_str())
        else {
            continue;
        };
        if !seen.insert(target_name.clone()) {
            return Err(validate_error(format!(
                "parakeet-ctc import mapped duplicate destination tensor '{target_name}'"
            )));
        }
        let target_dims = normalize_parakeet_weight_dims(&target_name, tensor.shape.as_slice());
        let data = safetensors.tensor_data(tensor)?;
        let tensor_type = quantized_tensor_type_for_parakeet_tensor(
            &target_name,
            &target_dims,
            force_f32,
            quantization,
        );
        let write_tensor = match tensor_type {
            Some(qtype) => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let quantized = quantize_f32_to_ggml_tensor_data(qtype, &target_dims, &values)
                    .map_err(|error| {
                        validate_error(format!(
                            "parakeet-ctc quantization failed for '{}' -> '{target_name}' ({qtype:?}): {error}",
                            tensor.name
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
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
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
    Ok(out)
}

/// `enc.blk.{i}.{suffix}` — the conformer encoder-layer convention shared with
/// the cohere packs (consumed by `nn::encoder::conformer_block`).
fn enc_blk(layer: usize, suffix: &str) -> String {
    format!("enc.blk.{layer}.{suffix}")
}

/// Map a parakeet HF tensor name to its `.oasr` target name + whether it must be
/// stored f32 (norms/biases/conv/subsampling/head; the linear projections may be
/// quantized). Returns `None` to drop a tensor (e.g. `num_batches_tracked`).
fn remap_parakeet_tensor_name(source_name: &str) -> Option<(String, bool)> {
    if source_name == "ctc_head.weight" {
        return Some(("ctc.head.weight".to_string(), false));
    }
    if source_name == "ctc_head.bias" {
        return Some(("ctc.head.bias".to_string(), true));
    }
    if let Some(rest) = source_name.strip_prefix("encoder.subsampling.") {
        // dw-striding subsampling: conv2d/linear weights stored source-dims, f32,
        // and consumed by the per-family prelude (S3), not the shared block.
        return Some((format!("enc.sub.{rest}"), true));
    }
    let rest = source_name.strip_prefix("encoder.layers.")?;
    let (layer, tail) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let suffix = match tail {
        "norm_feed_forward1.weight" => "ff1.norm.weight",
        "norm_feed_forward1.bias" => "ff1.norm.bias",
        "feed_forward1.linear1.weight" => "ff1.up.weight",
        "feed_forward1.linear1.bias" => "ff1.up.bias",
        "feed_forward1.linear2.weight" => "ff1.down.weight",
        "feed_forward1.linear2.bias" => "ff1.down.bias",
        "norm_self_att.weight" => "attn.norm.weight",
        "norm_self_att.bias" => "attn.norm.bias",
        "self_attn.q_proj.weight" => "attn.q.weight",
        "self_attn.q_proj.bias" => "attn.q.bias",
        "self_attn.k_proj.weight" => "attn.k.weight",
        "self_attn.k_proj.bias" => "attn.k.bias",
        "self_attn.v_proj.weight" => "attn.v.weight",
        "self_attn.v_proj.bias" => "attn.v.bias",
        "self_attn.o_proj.weight" => "attn.out.weight",
        "self_attn.o_proj.bias" => "attn.out.bias",
        "self_attn.relative_k_proj.weight" => "attn.pos.weight",
        "self_attn.bias_u" => "attn.pos_bias_u",
        "self_attn.bias_v" => "attn.pos_bias_v",
        "norm_conv.weight" => "conv.norm.weight",
        "norm_conv.bias" => "conv.norm.bias",
        "conv.pointwise_conv1.weight" => "conv.pw1.weight",
        "conv.pointwise_conv1.bias" => "conv.pw1.bias",
        "conv.depthwise_conv.weight" => "conv.dw.weight",
        "conv.depthwise_conv.bias" => "conv.dw.bias",
        "conv.norm.weight" => "conv.bn.weight",
        "conv.norm.bias" => "conv.bn.bias",
        "conv.norm.running_mean" => "conv.bn.mean",
        "conv.norm.running_var" => "conv.bn.var",
        "conv.pointwise_conv2.weight" => "conv.pw2.weight",
        "conv.pointwise_conv2.bias" => "conv.pw2.bias",
        "norm_feed_forward2.weight" => "ff2.norm.weight",
        "norm_feed_forward2.bias" => "ff2.norm.bias",
        "feed_forward2.linear1.weight" => "ff2.up.weight",
        "feed_forward2.linear1.bias" => "ff2.up.bias",
        "feed_forward2.linear2.weight" => "ff2.down.weight",
        "feed_forward2.linear2.bias" => "ff2.down.bias",
        "norm_out.weight" => "out.norm.weight",
        "norm_out.bias" => "out.norm.bias",
        "conv.norm.num_batches_tracked" => return None,
        _ => return None,
    };
    let target = enc_blk(layer, suffix);
    let force_f32 = parakeet_tensor_is_f32(&target);
    Some((target, force_f32))
}

/// f32-required tensors: norms, biases, the depthwise + BatchNorm conv tensors,
/// the subsampling prelude, and the CTC head. Only the 2-D linear projections
/// (`enc.blk.*.{ff,attn}.*.weight`, `attn.pos.weight`) may be quantized.
fn parakeet_tensor_is_f32(target_name: &str) -> bool {
    target_name.ends_with(".bias")
        || target_name.contains(".norm.")
        || target_name.contains(".bn.")
        || target_name.contains("conv.dw")
        || target_name.contains("conv.pw")
        || target_name.starts_with("enc.sub.")
        || target_name.starts_with("ctc.head")
}

/// Reverse the dims of 2-D+ projection weights (HF `[out, in]` → ggml `[in, out]`
/// for `mul_mat`), matching cohere. `pos_bias_u/v` ([heads, head_dim]) and conv
/// kernels are NOT projections and keep their source dims.
fn normalize_parakeet_weight_dims(target_name: &str, source_shape: &[u64]) -> Vec<u64> {
    if should_reverse_parakeet_tensor_dims(target_name) && source_shape.len() >= 2 {
        let mut dims = source_shape.to_vec();
        dims.reverse();
        dims
    } else {
        source_shape.to_vec()
    }
}

fn should_reverse_parakeet_tensor_dims(target_name: &str) -> bool {
    // EXACTLY cohere's rule (so every tensor feeding the shared conformer_block /
    // ggml conv matches its proven layout): reverse every rank>=2 `.weight` —
    // HF `[out, in]` linears -> ggml `[in, out]` for mul_mat, and HF conv kernels
    // `[OC, IC, kh, kw]` -> ggml `[kw, kh, IC, OC]` for conv_2d/conv_2d_dw — plus
    // the Transformer-XL `pos_bias_u/v` ([heads, head_dim] -> [head_dim, heads]).
    // 1-D biases/norms (len < 2) are untouched by `normalize_parakeet_weight_dims`.
    target_name.ends_with(".weight") || target_name.contains("pos_bias")
}

fn quantized_tensor_type_for_parakeet_tensor(
    name: &str,
    dims: &[u64],
    force_f32: bool,
    quantization: ParakeetCtcQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if force_f32 || quantization == ParakeetCtcQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    if !ne0.is_multiple_of(32_u64) {
        return None;
    }
    if quantization == ParakeetCtcQuantizationMode::Q4_K && ne0.is_multiple_of(256_u64) {
        return Some(GgufWriteTensorType::Q4_K);
    }
    Some(GgufWriteTensorType::Q8_0)
}

fn parakeet_runtime_gguf_metadata(
    config: &ParakeetConfigJson,
    request: &ParakeetCtcImportRequest,
    vocab_tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let enc = &config.encoder_config;
    let head_dim = enc.hidden_size / enc.num_attention_heads.max(1);
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str("general.architecture", PARAKEET_CTC_GGML_ARCHITECTURE_ID);
    // OASR v1 family-adapter selection metadata (required for runtime dispatch).
    put_str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put_str(OASR_METADATA_KEY_MODEL_FAMILY, PARAKEET_CTC_MODEL_FAMILY);
    put_str(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        PARAKEET_CTC_GGML_ARCHITECTURE_ID,
    );
    put_str(
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        PARAKEET_CTC_AUDIO_FRONTEND_ID,
    );
    put_str(
        OASR_METADATA_KEY_DECODE_POLICY,
        PARAKEET_CTC_DECODE_POLICY_ID,
    );
    put_str(GGML_TOKENIZER_ID_KEY, PARAKEET_CTC_TOKENIZER_ID);
    put_str("openasr.model.id", &request.model_id);

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32("parakeet.n_layers", enc.num_hidden_layers as u32);
    put_u32("parakeet.hidden_size", enc.hidden_size as u32);
    put_u32("parakeet.n_heads", enc.num_attention_heads as u32);
    put_u32("parakeet.head_dim", head_dim as u32);
    put_u32("parakeet.ffn_dim", enc.intermediate_size as u32);
    put_u32("parakeet.conv_kernel", enc.conv_kernel_size as u32);
    put_u32("parakeet.n_mels", enc.num_mel_bins as u32);
    put_u32("parakeet.subsampling_factor", enc.subsampling_factor as u32);
    put_u32(
        "parakeet.subsampling_channels",
        enc.subsampling_conv_channels as u32,
    );
    put_u32("parakeet.vocab_size", config.vocab_size as u32);
    put_u32("ctc.blank_token_id", config.pad_token_id);

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

    fn fixture_request() -> ParakeetCtcImportRequest {
        ParakeetCtcImportRequest {
            source_root: PathBuf::from("/tmp/parakeet-src"),
            output_root: PathBuf::from("/tmp/parakeet-out.oasr"),
            model_id: "parakeet-ctc-test".to_string(),
            quantization: ParakeetCtcQuantizationMode::Q8_0,
        }
    }

    fn fixture_config() -> ParakeetConfigJson {
        ParakeetConfigJson {
            encoder_config: ParakeetEncoderConfigJson {
                hidden_size: 16,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                intermediate_size: 32,
                conv_kernel_size: 9,
                num_mel_bins: 80,
                subsampling_factor: 8,
                subsampling_conv_channels: 16,
            },
            vocab_size: 4,
            pad_token_id: 3,
        }
    }

    fn string_metadata(metadata: &BTreeMap<String, GgufWriteValue>, key: &str) -> Option<String> {
        match metadata.get(key) {
            Some(GgufWriteValue::String(value)) => Some(value.clone()),
            _ => None,
        }
    }

    #[test]
    fn parakeet_runtime_metadata_declares_snapshot_streaming_feature() {
        let metadata = parakeet_runtime_gguf_metadata(
            &fixture_config(),
            &fixture_request(),
            &["<blank>".to_string(), "a".to_string(), "b".to_string()],
        );

        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            Some(PARAKEET_CTC_MODEL_FAMILY.to_string())
        );
        assert_eq!(
            string_metadata(&metadata, GGML_TOKENIZER_ID_KEY),
            Some(PARAKEET_CTC_TOKENIZER_ID.to_string())
        );
    }

    /// End-to-end round-trip on the real downloaded pack (skipped when absent so
    /// CI without host-local weights still passes). Gates the S2 importer: every
    /// expected conformer-layer tensor + subsampling + ctc head present, with
    /// projection dims reversed to ggml `[in, out]`, blank id 1024.
    ///
    /// Heavy (>60s): imports + quantizes the full multi-GB safetensors. Ignored by
    /// default to keep the suite fast; run on a host with the pack via
    /// `cargo nextest run --run-ignored all -E 'test(imports_parakeet_ctc_06b_round_trip)'`.
    #[test]
    #[ignore = "heavy: imports+quantizes a multi-GB model (>60s); run via --run-ignored"]
    fn imports_parakeet_ctc_06b_round_trip_when_pack_present() {
        let source_root = Path::new("../../tmp/models/parakeet-ctc-0.6b");
        let abs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/models/parakeet-ctc-0.6b");
        let source = if source_root.join("model.safetensors").exists() {
            source_root.to_path_buf()
        } else if abs.join("model.safetensors").exists() {
            abs
        } else {
            eprintln!("skipping: parakeet-ctc-0.6b pack not present");
            return;
        };

        let output = std::env::temp_dir().join("oasr_parakeet_ctc_roundtrip.oasr");
        let _ = std::fs::remove_file(&output);
        let result = convert_local_parakeet_ctc_source_to_runtime_pack(&ParakeetCtcImportRequest {
            source_root: source,
            output_root: output.clone(),
            model_id: "parakeet-ctc-0.6b-test".to_string(),
            quantization: ParakeetCtcQuantizationMode::Fp16,
        })
        .expect("parakeet import");
        assert_eq!(result.blank_token_id, 1024);
        crate::pull::preflight_model_pack_for_install(&output)
            .expect("imported pack must pass the generic pull preflight");

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

        // 24 conformer layers, each with the full ConformerBlockWeights set.
        for layer in 0..24 {
            for suffix in [
                "ff1.norm.weight",
                "ff1.up.weight",
                "ff1.down.weight",
                "attn.norm.weight",
                "attn.q.weight",
                "attn.k.weight",
                "attn.v.weight",
                "attn.out.weight",
                "attn.pos.weight",
                "attn.pos_bias_u",
                "attn.pos_bias_v",
                "conv.norm.weight",
                "conv.pw1.weight",
                "conv.dw.weight",
                "conv.bn.weight",
                "conv.bn.mean",
                "conv.bn.var",
                "conv.pw2.weight",
                "ff2.norm.weight",
                "ff2.up.weight",
                "ff2.down.weight",
                "out.norm.weight",
            ] {
                assert!(
                    names.contains(enc_blk(layer, suffix).as_str()),
                    "missing {layer}.{suffix}"
                );
            }
        }
        // Projection dims reversed to ggml [in, out].
        // Linear projections reversed HF [out,in] -> ggml [in,out].
        assert_eq!(dims_of(&enc_blk(0, "ff1.up.weight")), vec![1024, 4096]);
        assert_eq!(dims_of(&enc_blk(0, "ff1.down.weight")), vec![4096, 1024]);
        assert_eq!(dims_of(&enc_blk(0, "attn.q.weight")), vec![1024, 1024]);
        // pos_bias_u/v reversed [heads, head_dim] -> [head_dim, heads] (matches cohere).
        assert_eq!(dims_of(&enc_blk(0, "attn.pos_bias_u")), vec![128, 8]);
        // Subsampling conv0 reversed HF [OC,IC,kh,kw] -> ggml [kw,kh,IC,OC].
        assert!(names.contains("enc.sub.linear.weight"));
        assert_eq!(dims_of("enc.sub.layers.0.weight"), vec![3, 3, 1, 256]);
        // CTC head reversed; element count is vocab*hidden regardless of squeeze.
        assert_eq!(
            dims_of("ctc.head.weight").iter().product::<u64>(),
            1025 * 1024
        );
        assert_eq!(dims_of("ctc.head.bias"), vec![1025]);

        let _ = std::fs::remove_file(&output);
    }

    /// Producer (run with `--ignored`) that writes the canonical fp16 `.oasr` pack
    /// used by the S3 encoder smoke + S4 WER gate. Not a gate; just emits the file.
    #[test]
    #[ignore = "produces the host-local parakeet-ctc-0.6b .oasr pack"]
    fn produce_parakeet_ctc_06b_fp16_pack() {
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/models/parakeet-ctc-0.6b");
        let output = source.join("openasr/parakeet-ctc-0.6b-fp16.oasr");
        std::fs::create_dir_all(output.parent().unwrap()).unwrap();
        let result = convert_local_parakeet_ctc_source_to_runtime_pack(&ParakeetCtcImportRequest {
            source_root: source,
            output_root: output.clone(),
            model_id: "parakeet-ctc-0.6b".to_string(),
            quantization: ParakeetCtcQuantizationMode::Fp16,
        })
        .expect("parakeet fp16 pack");
        eprintln!(
            "wrote {} ({} tensors, blank {})",
            output.display(),
            result.tensor_count,
            result.blank_token_id
        );
    }
}
