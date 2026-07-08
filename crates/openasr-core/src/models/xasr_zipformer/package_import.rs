//! Convert a local `GilgameshWind/X-ASR-zh-en` source (icefall Zipformer2
//! transducer: `model.safetensors` + `config.json` + `tokens.txt`) into an
//! OpenASR `.oasr` (GGUF-v0) runtime pack.
//!
//! Unlike `parakeet_ctc::package_import` (which *remaps* onto the shared
//! `enc.blk.{i}.*` conformer convention), this importer is **name-preserving**:
//! every upstream icefall `state_dict` tensor is written under its exact name
//! (e.g. `encoder.encoders.3.encoder.layers.2.feed_forward1.in_proj.weight`).
//! The only transform is the proven cohere/parakeet dim-reversal of rank>=2
//! `.weight` tensors (HF `[out, in]` -> ggml `[in, out]` for `mul_mat`; conv
//! kernels `[OC, IC, kh, kw]` -> `[kw, kh, IC, OC]`), decided BEFORE quantization.
//! The `xasr_zipformer` executor (later stage) consumes these names directly.
//!
//! The normal source route is checkpoint/HF safetensors -> canonical
//! safetensors -> `.oasr`. X-ASR's deployed ONNX weights are a deliberate
//! exception because the public `.pt` is not numerically equivalent to the
//! 480 ms streaming deployment; `onnx_to_safetensors.py --xasr-remap`
//! normalizes that ONNX-authoritative source back into the same canonical
//! safetensors contract consumed here.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Deserialize;

use crate::arch::{
    XASR_ZIPFORMER_AUDIO_FRONTEND_ID, XASR_ZIPFORMER_DECODE_POLICY_ID,
    XASR_ZIPFORMER_GGML_ARCHITECTURE_ID, XASR_ZIPFORMER_MODEL_FAMILY, XASR_ZIPFORMER_TOKENIZER_ID,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, write_gguf_file_v0,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f16_bits,
    decode_safetensors_payload_as_f32, encode_f16_bits_le, read_source_file_bytes,
    read_source_json_file, validate_error, validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::pack_quant::{PackQuant, classify_quant_tensor};

const SOURCE_CONFIG_JSON: &str = "config.json";
const SOURCE_TOKENS_TXT: &str = "tokens.txt";
const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";

/// GGML caps tensor names at `GGML_MAX_NAME` (64 bytes incl. the NUL); names
/// longer than this are silently truncated on write and then fail the
/// type-set lookup. Keep compacted names within 63 bytes.
const GGUF_MAX_TENSOR_NAME: usize = 63;

/// Deterministic, collision-free compaction of icefall Zipformer2 `state_dict`
/// names (which run up to ~84 chars) into <=63-byte GGUF tensor names. The
/// `xasr_zipformer` executor MUST resolve tensors through this SAME function —
/// it is the shared name contract between the pack and the runtime. Verified
/// (see tests) to map all 989 X-ASR tensors to <=26 bytes with no collisions.
pub(crate) fn compact_xasr_name(name: &str) -> String {
    // Ordered longest/most-specific first so `self_attn_weights.` is abbreviated
    // before `self_attn.`, and the stack prefix before `encoder_embed.`.
    const REPLACEMENTS: &[(&str, &str)] = &[
        ("encoder.encoders.", "E"),
        (".encoder.layers.", ".L"),
        (".layers.", ".L"),
        ("encoder_embed.", "EE."),
        ("conv_module", "CM"),
        ("depthwise_conv.", "DW."),
        ("pointwise_conv", "PW"),
        ("chunkwise_conv", "KC"),
        ("causal_conv", "CC"),
        ("nonlin_attention.", "NA."),
        ("self_attn_weights.", "SAW."),
        ("self_attn.", "SA."),
        ("feed_forward", "FF"),
        ("in_proj", "IP"),
        ("out_proj", "OP"),
        ("linear_pos", "LP"),
        ("bypass", "BY"),
        ("downsample", "DS"),
        ("convnext", "CX"),
        ("output_linear", "OL"),
        ("embedding", "EMB"),
    ];
    let mut out = name.to_string();
    for (from, to) in REPLACEMENTS {
        if out.contains(from) {
            out = out.replace(from, to);
        }
    }
    out
}

pub type XasrZipformerQuantizationMode = PackQuant;

#[derive(Debug, Clone)]
pub struct XasrZipformerImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: XasrZipformerQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XasrZipformerImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub blank_token_id: u32,
}

/// The Zipformer2 architecture config. These are per-stack arrays (6 stacks for
/// X-ASR) plus scalar decode/joiner/tokenizer dims. Values mirror the ONNX
/// `metadata_props` (`num_encoder_layers`, `encoder_dims`, ...). `config.json`
/// is authored alongside the safetensors source (the upstream HF `config.json`
/// is too thin), so this importer reads the architecture from it rather than
/// hard-coding a single checkpoint's shape.
#[derive(Debug, Deserialize)]
struct XasrZipformerConfigJson {
    num_encoder_layers: Vec<u32>,
    encoder_dims: Vec<u32>,
    query_head_dims: Vec<u32>,
    value_head_dims: Vec<u32>,
    num_heads: Vec<u32>,
    cnn_module_kernels: Vec<u32>,
    left_context_len: Vec<u32>,
    downsampling_factors: Vec<u32>,
    feature_dim: u32,
    decode_chunk_len: u32,
    joiner_dim: u32,
    decoder_context_size: u32,
    vocab_size: u32,
    blank_id: u32,
}

pub fn convert_local_xasr_zipformer_source_to_runtime_pack(
    request: &XasrZipformerImportRequest,
) -> Result<XasrZipformerImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_root)?;
    let config: XasrZipformerConfigJson =
        read_source_json_file(&request.source_root, SOURCE_CONFIG_JSON)?;
    validate_config(&config)?;
    let vocab_tokens = parse_tokens_txt(&request.source_root, config.vocab_size)?;
    let model_path = request.source_root.join(SOURCE_MODEL_SAFETENSORS);
    let safetensors = SafetensorsFile::open(&model_path)?;

    let blank_token_id = config.blank_id;
    let tensors = build_xasr_runtime_tensors(&safetensors, request.quantization)?;
    let metadata = xasr_runtime_gguf_metadata(&config, request, &vocab_tokens);

    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "xasr-zipformer GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "xasr-zipformer import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(XasrZipformerImportResult {
        output_path: request.output_root.clone(),
        tensor_count: index.tensors().len(),
        blank_token_id,
    })
}

fn validate_config(config: &XasrZipformerConfigJson) -> Result<(), LocalSourceImportError> {
    let stacks = config.num_encoder_layers.len();
    if stacks == 0 {
        return Err(validate_error(
            "xasr-zipformer config num_encoder_layers must not be empty",
        ));
    }
    for (field, len) in [
        ("encoder_dims", config.encoder_dims.len()),
        ("query_head_dims", config.query_head_dims.len()),
        ("value_head_dims", config.value_head_dims.len()),
        ("num_heads", config.num_heads.len()),
        ("cnn_module_kernels", config.cnn_module_kernels.len()),
        ("left_context_len", config.left_context_len.len()),
        ("downsampling_factors", config.downsampling_factors.len()),
    ] {
        if len != stacks {
            return Err(validate_error(format!(
                "xasr-zipformer config '{field}' has {len} entries, expected {stacks} (one per stack)"
            )));
        }
    }
    if config.vocab_size == 0 {
        return Err(validate_error(
            "xasr-zipformer config vocab_size must be > 0",
        ));
    }
    if config.blank_id >= config.vocab_size {
        return Err(validate_error(format!(
            "xasr-zipformer config blank_id {} is out of range for vocab_size {}",
            config.blank_id, config.vocab_size
        )));
    }
    Ok(())
}

/// Parse a k2/icefall `tokens.txt` (`<symbol> <id>` per line) into an ordered,
/// id-indexed token list of length `vocab_size`.
fn parse_tokens_txt(
    source_root: &std::path::Path,
    vocab_size: u32,
) -> Result<Vec<String>, LocalSourceImportError> {
    let bytes = read_source_file_bytes(source_root, SOURCE_TOKENS_TXT)?;
    let text = String::from_utf8(bytes)
        .map_err(|_| validate_error("xasr-zipformer tokens.txt is not valid UTF-8"))?;
    let vocab = vocab_size as usize;
    let mut tokens = vec![None::<String>; vocab];
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let (symbol, id_str) = line.rsplit_once(' ').ok_or_else(|| {
            validate_error(format!(
                "xasr-zipformer tokens.txt line {} is not '<symbol> <id>': {line:?}",
                line_no + 1
            ))
        })?;
        let id: usize = id_str.trim().parse().map_err(|_| {
            validate_error(format!(
                "xasr-zipformer tokens.txt line {} has a non-integer id {id_str:?}",
                line_no + 1
            ))
        })?;
        if id < vocab {
            tokens[id] = Some(symbol.to_string());
        }
    }
    tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| {
            token.ok_or_else(|| {
                validate_error(format!(
                    "xasr-zipformer tokens.txt is missing token for id {id}"
                ))
            })
        })
        .collect()
}

fn build_xasr_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: XasrZipformerQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tensor in &safetensors.header().tensors {
        let original = tensor.name.as_str();
        // Compact the long icefall name to fit GGML's 64-byte tensor-name cap;
        // the executor resolves tensors through the same `compact_xasr_name`.
        // Predicates (dim-reversal, f32, quant) run on the ORIGINAL name.
        let target_name = compact_xasr_name(original);
        if target_name.len() > GGUF_MAX_TENSOR_NAME {
            return Err(validate_error(format!(
                "xasr-zipformer compacted name '{target_name}' ({} bytes) exceeds the GGUF \
                 {GGUF_MAX_TENSOR_NAME}-byte limit (from '{original}')",
                target_name.len()
            )));
        }
        if !seen.insert(target_name.clone()) {
            return Err(validate_error(format!(
                "xasr-zipformer name compaction collided on '{target_name}' (from '{original}')"
            )));
        }
        let target_dims = normalize_xasr_weight_dims(original, tensor.shape.as_slice());
        let force_f32 = xasr_tensor_is_f32(original, &target_dims);
        let data = safetensors.tensor_data(tensor)?;
        let tensor_type =
            quantized_tensor_type_for_xasr_tensor(original, &target_dims, force_f32, quantization);
        let write_tensor = match tensor_type {
            Some(qtype) => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let quantized = quantize_f32_to_ggml_tensor_data(qtype, &target_dims, &values)
                    .map_err(|error| {
                        validate_error(format!(
                            "xasr-zipformer quantization failed for '{}' ({qtype:?}): {error}",
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
    if out.is_empty() {
        return Err(validate_error("xasr-zipformer source produced no tensors"));
    }
    Ok(out)
}

/// Reverse the dims of rank>=2 `.weight` tensors (HF `[out, in]` -> ggml
/// `[in, out]` for `mul_mat`; conv kernels `[OC, IC, k...]` -> `[k..., IC, OC]`),
/// exactly as cohere/parakeet do. Scales (`*_scale`), biases, and 1-D tensors
/// keep their source dims.
fn normalize_xasr_weight_dims(target_name: &str, source_shape: &[u64]) -> Vec<u64> {
    if target_name.ends_with(".weight") && source_shape.len() >= 2 {
        let mut dims = source_shape.to_vec();
        dims.reverse();
        dims
    } else {
        source_shape.to_vec()
    }
}

/// f32-required tensors: biases, every norm/scale (BiasNorm `log_scale`, bypass
/// scales, `chunkwise_conv_scale`), the bypass projections, the embedding table
/// (consumed by ggml `get_rows`), and anything that is not a plain 2-D matrix
/// (conv kernels, 1-D params). Only rank-2 `.weight` projection matrices may be
/// quantized.
fn xasr_tensor_is_f32(name: &str, dims: &[u64]) -> bool {
    name.ends_with(".bias")
        || name.contains("norm")
        || name.contains("_scale")
        || name.contains("bypass")
        || name == "decoder.embedding.weight"
        || dims.len() != 2
}

fn quantized_tensor_type_for_xasr_tensor(
    name: &str,
    dims: &[u64],
    force_f32: bool,
    quantization: XasrZipformerQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if force_f32 || quantization == XasrZipformerQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    classify_quant_tensor(ne0, quantization)
}

fn join_u32(values: &[u32]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn xasr_runtime_gguf_metadata(
    config: &XasrZipformerConfigJson,
    request: &XasrZipformerImportRequest,
    vocab_tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str("general.architecture", XASR_ZIPFORMER_GGML_ARCHITECTURE_ID);
    // OASR v1 family-adapter selection metadata (required for runtime dispatch).
    put_str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put_str(OASR_METADATA_KEY_MODEL_FAMILY, XASR_ZIPFORMER_MODEL_FAMILY);
    put_str(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
    );
    put_str(
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        XASR_ZIPFORMER_AUDIO_FRONTEND_ID,
    );
    put_str(
        OASR_METADATA_KEY_DECODE_POLICY,
        XASR_ZIPFORMER_DECODE_POLICY_ID,
    );
    put_str(GGML_TOKENIZER_ID_KEY, XASR_ZIPFORMER_TOKENIZER_ID);
    put_str("openasr.model.id", &request.model_id);

    // Per-stack architecture arrays are comma-joined strings (matches the ONNX
    // metadata_props format); the executor splits on ',' at load.
    put_str(
        "xasr.num_encoder_layers",
        &join_u32(&config.num_encoder_layers),
    );
    put_str("xasr.encoder_dims", &join_u32(&config.encoder_dims));
    put_str("xasr.query_head_dims", &join_u32(&config.query_head_dims));
    put_str("xasr.value_head_dims", &join_u32(&config.value_head_dims));
    put_str("xasr.num_heads", &join_u32(&config.num_heads));
    put_str(
        "xasr.cnn_module_kernels",
        &join_u32(&config.cnn_module_kernels),
    );
    put_str("xasr.left_context_len", &join_u32(&config.left_context_len));
    put_str(
        "xasr.downsampling_factors",
        &join_u32(&config.downsampling_factors),
    );

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32("xasr.num_stacks", config.num_encoder_layers.len() as u32);
    put_u32("xasr.feature_dim", config.feature_dim);
    put_u32("xasr.decode_chunk_len", config.decode_chunk_len);
    put_u32("xasr.joiner_dim", config.joiner_dim);
    put_u32("xasr.decoder_context_size", config.decoder_context_size);
    put_u32("xasr.vocab_size", config.vocab_size);
    put_u32("xasr.blank_id", config.blank_id);

    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufWriteValue::StringArray(vocab_tokens.to_vec()),
    );
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::read_gguf_metadata;
    use crate::models::xasr_zipformer::runtime_contract::parse_xasr_zipformer_execution_metadata;
    use std::path::Path;

    fn fixture_config() -> XasrZipformerConfigJson {
        XasrZipformerConfigJson {
            num_encoder_layers: vec![2, 2, 4, 5, 4, 2],
            encoder_dims: vec![192, 256, 512, 768, 512, 256],
            query_head_dims: vec![32, 32, 32, 32, 32, 32],
            value_head_dims: vec![12, 12, 12, 12, 12, 12],
            num_heads: vec![4, 4, 4, 8, 4, 4],
            cnn_module_kernels: vec![31, 31, 15, 15, 15, 31],
            left_context_len: vec![256, 128, 64, 32, 64, 128],
            downsampling_factors: vec![1, 2, 4, 8, 4, 2],
            feature_dim: 80,
            decode_chunk_len: 48,
            joiner_dim: 512,
            decoder_context_size: 2,
            vocab_size: 5000,
            blank_id: 0,
        }
    }

    fn fixture_request() -> XasrZipformerImportRequest {
        XasrZipformerImportRequest {
            source_root: PathBuf::from("/tmp/xasr-src"),
            output_root: PathBuf::from("/tmp/xasr-out.oasr"),
            model_id: "x-asr-zh-en-test".to_string(),
            quantization: XasrZipformerQuantizationMode::Fp16,
        }
    }

    fn string_metadata(metadata: &BTreeMap<String, GgufWriteValue>, key: &str) -> Option<String> {
        match metadata.get(key) {
            Some(GgufWriteValue::String(value)) => Some(value.clone()),
            _ => None,
        }
    }

    #[test]
    fn metadata_declares_streaming_family_and_stack_arrays() {
        let metadata = xasr_runtime_gguf_metadata(
            &fixture_config(),
            &fixture_request(),
            &["<blk>".to_string()],
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            Some(XASR_ZIPFORMER_MODEL_FAMILY.to_string())
        );
        assert_eq!(
            string_metadata(&metadata, "xasr.num_encoder_layers"),
            Some("2,2,4,5,4,2".to_string())
        );
        assert_eq!(
            string_metadata(&metadata, "xasr.encoder_dims"),
            Some("192,256,512,768,512,256".to_string())
        );
        assert_eq!(
            string_metadata(&metadata, "xasr.left_context_len"),
            Some("256,128,64,32,64,128".to_string())
        );
        assert_eq!(
            metadata.get("xasr.decode_chunk_len"),
            Some(&GgufWriteValue::U32(48))
        );
    }

    #[test]
    fn weight_dims_reverse_only_rank2plus_weight() {
        // rank-2 linear: HF [out, in] -> ggml [in, out]
        assert_eq!(
            normalize_xasr_weight_dims("joiner.output_linear.weight", &[5000, 512]),
            vec![512, 5000]
        );
        // conv kernel [OC, IC, k] -> reversed
        assert_eq!(
            normalize_xasr_weight_dims("decoder.conv.weight", &[512, 4, 2]),
            vec![2, 4, 512]
        );
        // scale (not .weight) keeps source dims
        assert_eq!(
            normalize_xasr_weight_dims(
                "encoder.encoders.0.layers.0.conv_module1.depthwise_conv.chunkwise_conv_scale",
                &[2, 192, 31]
            ),
            vec![2, 192, 31]
        );
        // 1-D bias untouched
        assert_eq!(
            normalize_xasr_weight_dims("encoder_proj.bias", &[512]),
            vec![512]
        );
    }

    #[test]
    fn quant_policy_keeps_sensitive_tensors_f32() {
        // embedding -> f32 (get_rows source), even in q8 mode
        assert!(xasr_tensor_is_f32("decoder.embedding.weight", &[512, 5000]));
        // scales / norms / bias / bypass -> f32
        assert!(xasr_tensor_is_f32(
            "encoder.encoders.0.layers.0.bypass.bypass_scale",
            &[192]
        ));
        assert!(xasr_tensor_is_f32(
            "encoder.encoders.0.layers.0.feed_forward1.out_proj.bias",
            &[192]
        ));
        // a plain rank-2 projection weight is quantizable in q8 mode
        let q = quantized_tensor_type_for_xasr_tensor(
            "encoder.encoders.0.layers.0.feed_forward1.in_proj.weight",
            &[192, 384],
            false,
            XasrZipformerQuantizationMode::Q8_0,
        );
        assert_eq!(q, Some(GgufWriteTensorType::Q8_0));
    }

    #[test]
    fn name_compaction_is_short_and_structure_preserving() {
        assert_eq!(
            compact_xasr_name(
                "encoder.encoders.3.encoder.layers.2.conv_module1.depthwise_conv.chunkwise_conv_scale"
            ),
            "E3.L2.CM1.DW.KC_scale"
        );
        assert_eq!(
            compact_xasr_name("encoder.encoders.0.layers.0.feed_forward1.in_proj.weight"),
            "E0.L0.FF1.IP.weight"
        );
        assert_eq!(
            compact_xasr_name("joiner.output_linear.weight"),
            "joiner.OL.weight"
        );
        assert_eq!(
            compact_xasr_name("decoder.embedding.weight"),
            "decoder.EMB.weight"
        );
        // Top-level (non-stack) names that have no abbreviation pass through.
        assert_eq!(
            compact_xasr_name("simple_am_proj.weight"),
            "simple_am_proj.weight"
        );
        assert!(
            compact_xasr_name(
                "encoder.encoders.3.encoder.layers.4.nonlin_attention.out_proj.weight"
            )
            .len()
                <= GGUF_MAX_TENSOR_NAME
        );
    }

    #[test]
    fn tokens_txt_parses_id_indexed() {
        let dir = std::env::temp_dir().join("xasr-tokens-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokens.txt"),
            "<blk> 0\n<sos/eos> 1\n\u{2581}a 2\n",
        )
        .unwrap();
        let tokens = parse_tokens_txt(&dir, 3).unwrap();
        assert_eq!(tokens, vec!["<blk>", "<sos/eos>", "\u{2581}a"]);
    }

    /// End-to-end round-trip on the real downloaded pack (skipped when absent so
    /// CI without host-local weights still passes). Imports the staged
    /// `tmp/xasr-test/src` (produced by `pt_to_safetensors.py`) and asserts the
    /// full 989-tensor set round-trips with reversed projection dims.
    ///
    /// Heavy (>30s): run via
    /// `cargo nextest run -p openasr-core --run-ignored all -E 'test(imports_xasr_zipformer_round_trip)'`.
    #[test]
    #[ignore = "heavy: imports a multi-GB model; run via --run-ignored"]
    fn imports_xasr_zipformer_round_trip_when_pack_present() {
        let rel = Path::new("../../tmp/xasr-test/src");
        let abs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/src");
        let source = if rel.join("model.safetensors").exists() {
            rel.to_path_buf()
        } else if abs.join("model.safetensors").exists() {
            abs
        } else {
            eprintln!("skipping: xasr-test/src/model.safetensors not present");
            return;
        };
        let output = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-fp16.oasr");
        // The writer fail-closes on an existing path; clear any prior run's pack.
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let _ = std::fs::remove_file(&output);
        let request = XasrZipformerImportRequest {
            source_root: source,
            output_root: output.clone(),
            model_id: "x-asr-zh-en".to_string(),
            quantization: XasrZipformerQuantizationMode::Fp16,
        };
        let result = convert_local_xasr_zipformer_source_to_runtime_pack(&request)
            .expect("import should succeed");
        assert_eq!(result.tensor_count, 989, "all 989 icefall tensors present");
        assert_eq!(result.blank_token_id, 0);
        crate::pull::preflight_model_pack_for_install(&output)
            .expect("imported pack must pass the generic pull preflight");

        let index = read_gguf_tensor_index(&output).expect("readable index");
        let joiner = index
            .get("joiner.OL.weight")
            .expect("joiner output present (compacted name)");
        assert_eq!(
            joiner.dims,
            vec![512, 5000],
            "HF [5000,512] reversed to ggml [in,out]"
        );
        let embed = index
            .get("decoder.EMB.weight")
            .expect("embedding present (compacted name)");
        assert_eq!(embed.dims, vec![512, 5000]);
        assert_eq!(embed.type_name, "f32", "embedding kept f32 for get_rows");
        // Every stored name must respect the GGUF 64-byte cap.
        for tensor in index.tensors() {
            assert!(
                tensor.name.len() <= GGUF_MAX_TENSOR_NAME,
                "tensor name too long: {} ({} bytes)",
                tensor.name,
                tensor.name.len()
            );
        }
    }

    /// Same importer path, but using ONNX deployment initializers normalized by
    /// `tooling/publish-model/scripts/onnx_to_safetensors.py --xasr-remap`.
    /// This is the quality-parity source for `GilgameshWind/X-ASR-zh-en`; the
    /// public `streaming_exp/pretrained.pt` does not numerically match the ONNX
    /// deployment artifacts.
    #[test]
    #[ignore = "heavy: imports ONNX-derived X-ASR safetensors; run via --run-ignored"]
    fn imports_xasr_zipformer_onnx_round_trip_when_pack_present() {
        let abs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/src-onnx");
        if !abs.join("model.safetensors").exists() {
            eprintln!("skipping: xasr-test/src-onnx/model.safetensors not present");
            return;
        }
        let output = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-fp16.oasr");
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let _ = std::fs::remove_file(&output);
        let request = XasrZipformerImportRequest {
            source_root: abs,
            output_root: output.clone(),
            model_id: "x-asr-zh-en-onnx".to_string(),
            quantization: XasrZipformerQuantizationMode::Fp16,
        };
        let result = convert_local_xasr_zipformer_source_to_runtime_pack(&request)
            .expect("ONNX-derived import should succeed");
        assert_eq!(result.tensor_count, 966);
        assert_eq!(result.blank_token_id, 0);

        let index = read_gguf_tensor_index(&output).expect("readable index");
        let decoder_proj = index
            .get("joiner.decoder_proj.weight")
            .expect("ONNX decoder_proj remapped into joiner namespace");
        assert_eq!(decoder_proj.dims, vec![512, 512]);
        let encoder_proj = index
            .get("joiner.encoder_proj.weight")
            .expect("ONNX encoder_proj remapped into joiner namespace");
        assert_eq!(encoder_proj.dims, vec![768, 512]);
        let output_linear = index
            .get("joiner.OL.weight")
            .expect("ONNX output_linear remapped into joiner namespace");
        assert_eq!(output_linear.dims, vec![512, 5000]);
        let metadata = read_gguf_metadata(&output).expect("readable metadata");
        let metadata =
            parse_xasr_zipformer_execution_metadata(&metadata).expect("xasr metadata parses");
        assert_eq!(metadata.decode_chunk_len, 48);
        assert_eq!(metadata.left_context_len, vec![256, 128, 64, 32, 64, 128]);
        let scale = index
            .get(&compact_xasr_name(
                "encoder.encoders.0.layers.0.conv_module1.depthwise_conv.chunkwise_conv_scale",
            ))
            .expect("ONNX anonymous chunkwise edge scales remapped to semantic tensor");
        assert_eq!(scale.dims, vec![2, 192, 31]);
        assert_eq!(scale.type_name, "f32");
        let downsample = index
            .get(&compact_xasr_name("encoder.encoders.1.downsample.bias"))
            .expect("ONNX anonymous downsample weights remapped to semantic bias logits");
        assert_eq!(downsample.dims, vec![2]);
        assert_eq!(downsample.type_name, "f32");
        let downsample_output = index
            .get(&compact_xasr_name("encoder.downsample_output.bias"))
            .expect("ONNX anonymous output downsample weights remapped to semantic bias logits");
        assert_eq!(downsample_output.dims, vec![2]);
        assert_eq!(downsample_output.type_name, "f32");
        assert!(
            index
                .tensors()
                .iter()
                .all(|tensor| !tensor.name.starts_with("onnx::Slice_")
                    && !tensor.name.starts_with("onnx::Mul_")),
            "anonymous ONNX Slice/Mul initializers must not leak into runtime pack"
        );
    }
}
