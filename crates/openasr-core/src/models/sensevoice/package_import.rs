//! Convert a local `FunAudioLLM/SenseVoiceSmall` FunASR source (a
//! `model.safetensors` produced by `pt_to_safetensors.py` from the checkpoint's
//! `model.pt`, plus `am.mvn`, `config.yaml`, and the SentencePiece
//! `chn_jpn_yue_eng_ko_spectok.bpe.model`) into an OpenASR `.oasr` (GGUF-v0)
//! runtime pack.
//!
//! Mirrors `parakeet_ctc::package_import` (the same safetensors -> GGUF path).
//! Layer mapping: `encoder.encoders0.0` (the 560-dim input layer) becomes
//! `enc.blk.0`, `encoder.encoders.{i}` becomes `enc.blk.{i+1}` (50 SAN-M blocks
//! total), and `encoder.tp_encoders.{i}` becomes `tp.blk.{i}` (20 blocks). The
//! CMVN vectors are parsed out of the checkpoint's kaldi-text `am.mvn` and baked
//! as `frontend.cmvn.*` f32 tensors; the SentencePiece vocab pieces are parsed
//! from the binary model proto and embedded as `tokenizer.ggml.tokens`.
//!
//! Keep-quantized: 2-D linear projections (`attn.qkv/out`, `ffn.up/down`,
//! `ctc.head.weight`) quantize to q8_0/q4_k; norms, biases, the FSMN depthwise
//! kernels, the CMVN vectors, and the 16x560 prompt-embedding table stay
//! f32 (they are not `mul_mat` weights).

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::arch::{
    SENSEVOICE_AUDIO_FRONTEND_ID, SENSEVOICE_DECODE_POLICY_ID, SENSEVOICE_TOKENIZER_ID,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, write_gguf_file_v0,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f16_bits,
    decode_safetensors_payload_as_f32, encode_f16_bits_le, validate_error,
    validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};
use crate::models::pack_quant::{PackQuant, classify_quant_tensor};

use crate::arch::{SENSEVOICE_GGML_ARCHITECTURE_ID, SENSEVOICE_MODEL_FAMILY};

const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";
const SOURCE_AM_MVN: &str = "am.mvn";
const SOURCE_CONFIG_YAML: &str = "config.yaml";
const SOURCE_SPM_MODEL: &str = "chn_jpn_yue_eng_ko_spectok.bpe.model";

/// FunASR's CTC blank id (piece 0, `<unk>`, doubles as the blank).
const SENSEVOICE_CTC_BLANK_ID: u32 = 0;

pub type SenseVoiceQuantizationMode = PackQuant;

#[derive(Debug, Clone)]
pub struct SenseVoiceImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: SenseVoiceQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SenseVoiceImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub vocab_size: usize,
}

/// Architecture facts derived from the safetensors shapes (fail-closed:
/// inconsistent shapes reject the import rather than writing a broken pack).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SenseVoiceDerivedHparams {
    n_layers: usize,
    tp_layers: usize,
    d_model: usize,
    n_heads: usize,
    ffn_dim: usize,
    fsmn_kernel: usize,
    feature_dim: usize,
    vocab_size: usize,
}

pub fn convert_local_sensevoice_source_to_runtime_pack(
    request: &SenseVoiceImportRequest,
) -> Result<SenseVoiceImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_root)?;
    let model_path = request.source_root.join(SOURCE_MODEL_SAFETENSORS);
    let safetensors = SafetensorsFile::open(&model_path)?;

    let (cmvn_neg_mean, cmvn_inv_stddev) = parse_am_mvn(&request.source_root.join(SOURCE_AM_MVN))?;
    let vocab_tokens = parse_sentencepiece_pieces(&request.source_root.join(SOURCE_SPM_MODEL))?;
    let n_heads = parse_attention_heads(&request.source_root.join(SOURCE_CONFIG_YAML))?;

    let hparams = derive_and_validate_hparams(&safetensors, n_heads, vocab_tokens.len())?;
    if cmvn_neg_mean.len() != hparams.feature_dim || cmvn_inv_stddev.len() != hparams.feature_dim {
        return Err(validate_error(format!(
            "sensevoice am.mvn dim {} does not match the model feature dim {}",
            cmvn_neg_mean.len(),
            hparams.feature_dim
        )));
    }

    let mut tensors = build_sensevoice_runtime_tensors(&safetensors, request.quantization)?;
    tensors.push(f32_tensor(
        "frontend.cmvn.neg_mean",
        vec![hparams.feature_dim as u64],
        &cmvn_neg_mean,
    ));
    tensors.push(f32_tensor(
        "frontend.cmvn.inv_stddev",
        vec![hparams.feature_dim as u64],
        &cmvn_inv_stddev,
    ));

    let metadata = sensevoice_runtime_gguf_metadata(&hparams, request, &vocab_tokens);
    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "sensevoice GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "sensevoice import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(SenseVoiceImportResult {
        output_path: request.output_root.clone(),
        tensor_count: index.tensors().len(),
        vocab_size: vocab_tokens.len(),
    })
}

fn f32_tensor(name: &str, dims: Vec<u64>, values: &[f32]) -> GgufWriteTensor {
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

/// Parse the two `<LearnRateCoef> 0 [ ... ]` vectors out of the kaldi-nnet text
/// `am.mvn`: the `<AddShift>` row is the negated mean and the `<Rescale>` row is
/// the inverse stddev (so normalization is `(x + neg_mean) * inv_stddev`).
fn parse_am_mvn(path: &Path) -> Result<(Vec<f32>, Vec<f32>), LocalSourceImportError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        validate_error(format!(
            "sensevoice import cannot read '{}': {error}",
            path.display()
        ))
    })?;
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let mut rest = text.as_str();
    while let Some(start) = rest.find("<LearnRateCoef>") {
        let after = &rest[start..];
        let Some(open) = after.find('[') else { break };
        let Some(close) = after.find(']') else {
            return Err(validate_error(
                "sensevoice am.mvn has an unterminated '[' vector".to_string(),
            ));
        };
        if close < open {
            return Err(validate_error(
                "sensevoice am.mvn has ']' before '['".to_string(),
            ));
        }
        let body = &after[open + 1..close];
        let values = body
            .split_whitespace()
            .map(|token| {
                token.parse::<f32>().map_err(|error| {
                    validate_error(format!(
                        "sensevoice am.mvn has a non-numeric value '{token}': {error}"
                    ))
                })
            })
            .collect::<Result<Vec<f32>, _>>()?;
        vectors.push(values);
        rest = &after[close + 1..];
    }
    if vectors.len() != 2 {
        return Err(validate_error(format!(
            "sensevoice am.mvn must contain exactly 2 CMVN vectors (AddShift + Rescale), found {}",
            vectors.len()
        )));
    }
    let inv_stddev = vectors.pop().expect("checked len");
    let neg_mean = vectors.pop().expect("checked len");
    if neg_mean.len() != inv_stddev.len() || neg_mean.is_empty() {
        return Err(validate_error(format!(
            "sensevoice am.mvn vector lengths differ or are empty ({} vs {})",
            neg_mean.len(),
            inv_stddev.len()
        )));
    }
    Ok((neg_mean, inv_stddev))
}

/// Extract the ordered vocab piece strings from a binary SentencePiece
/// `ModelProto` (field 1 = repeated `SentencePiece`; inside, field 1 = the piece
/// string). Minimal protobuf wire walk -- no proto dependency; fails closed on
/// malformed input.
fn parse_sentencepiece_pieces(path: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let data = std::fs::read(path).map_err(|error| {
        validate_error(format!(
            "sensevoice import cannot read '{}': {error}",
            path.display()
        ))
    })?;
    let malformed =
        |what: &str| validate_error(format!("sensevoice spm model is malformed: {what}"));

    fn read_varint(data: &[u8], mut i: usize) -> Option<(u64, usize)> {
        let mut value: u64 = 0;
        let mut shift = 0u32;
        loop {
            let byte = *data.get(i)?;
            i += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Some((value, i));
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
    }
    fn skip_field(data: &[u8], i: usize, wire: u64) -> Option<usize> {
        match wire {
            0 => read_varint(data, i).map(|(_, next)| next),
            1 => i.checked_add(8).filter(|&n| n <= data.len()),
            2 => {
                let (len, next) = read_varint(data, i)?;
                let len = usize::try_from(len).ok()?;
                next.checked_add(len).filter(|&n| n <= data.len())
            }
            5 => i.checked_add(4).filter(|&n| n <= data.len()),
            _ => None,
        }
    }

    let mut pieces = Vec::new();
    let mut i = 0usize;
    while i < data.len() {
        let (tag, next) = read_varint(&data, i).ok_or_else(|| malformed("truncated tag"))?;
        i = next;
        let (field, wire) = (tag >> 3, tag & 7);
        if field == 1 && wire == 2 {
            let (len, next) =
                read_varint(&data, i).ok_or_else(|| malformed("truncated piece length"))?;
            let len = usize::try_from(len).map_err(|_| malformed("oversized piece length"))?;
            let end = next
                .checked_add(len)
                .filter(|&n| n <= data.len())
                .ok_or_else(|| malformed("piece length exceeds file"))?;
            let sub = &data[next..end];
            i = end;
            // Parse the SentencePiece submessage; field 1 = piece string.
            let mut piece: Option<String> = None;
            let mut j = 0usize;
            while j < sub.len() {
                let (sub_tag, sub_next) =
                    read_varint(sub, j).ok_or_else(|| malformed("truncated sub tag"))?;
                j = sub_next;
                let (sub_field, sub_wire) = (sub_tag >> 3, sub_tag & 7);
                if sub_field == 1 && sub_wire == 2 {
                    let (sub_len, val_start) =
                        read_varint(sub, j).ok_or_else(|| malformed("truncated piece"))?;
                    let sub_len =
                        usize::try_from(sub_len).map_err(|_| malformed("oversized piece"))?;
                    let val_end = val_start
                        .checked_add(sub_len)
                        .filter(|&n| n <= sub.len())
                        .ok_or_else(|| malformed("piece exceeds message"))?;
                    piece = Some(
                        std::str::from_utf8(&sub[val_start..val_end])
                            .map_err(|_| malformed("piece is not valid UTF-8"))?
                            .to_string(),
                    );
                    j = val_end;
                } else {
                    j = skip_field(sub, j, sub_wire)
                        .ok_or_else(|| malformed("bad sub wire type"))?;
                }
            }
            pieces.push(piece.ok_or_else(|| malformed("SentencePiece without a piece string"))?);
        } else {
            i = skip_field(&data, i, wire).ok_or_else(|| malformed("bad wire type"))?;
        }
    }
    if pieces.is_empty() {
        return Err(malformed("no vocab pieces"));
    }
    Ok(pieces)
}

/// Read `attention_heads: N` from the checkpoint's `config.yaml` `encoder_conf`.
/// The head count is the one architecture fact not recoverable from tensor
/// shapes (QKV fuses the heads), so it must come from the config -- fail closed
/// if absent rather than guessing.
fn parse_attention_heads(path: &Path) -> Result<usize, LocalSourceImportError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        validate_error(format!(
            "sensevoice import cannot read '{}': {error}",
            path.display()
        ))
    })?;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("attention_heads:") {
            return rest.trim().parse::<usize>().map_err(|error| {
                validate_error(format!(
                    "sensevoice config.yaml has a non-numeric attention_heads: {error}"
                ))
            });
        }
    }
    Err(validate_error(
        "sensevoice config.yaml is missing encoder_conf.attention_heads".to_string(),
    ))
}

fn derive_and_validate_hparams(
    safetensors: &SafetensorsFile,
    n_heads: usize,
    vocab_pieces: usize,
) -> Result<SenseVoiceDerivedHparams, LocalSourceImportError> {
    let mut shape_by_name: BTreeMap<&str, &[u64]> = BTreeMap::new();
    for tensor in &safetensors.header().tensors {
        shape_by_name.insert(tensor.name.as_str(), tensor.shape.as_slice());
    }
    let shape = |name: &str| -> Result<&[u64], LocalSourceImportError> {
        shape_by_name
            .get(name)
            .copied()
            .ok_or_else(|| validate_error(format!("sensevoice source is missing tensor '{name}'")))
    };

    // Layer counts by scanning contiguous indices (fail-closed on gaps).
    let count_layers = |prefix: &str| -> usize {
        let mut count = 0usize;
        while shape_by_name
            .contains_key(format!("{prefix}.{count}.self_attn.linear_q_k_v.weight").as_str())
        {
            count += 1;
        }
        count
    };
    if !shape_by_name.contains_key("encoder.encoders0.0.self_attn.linear_q_k_v.weight") {
        return Err(validate_error(
            "sensevoice source is missing the encoders0 input layer".to_string(),
        ));
    }
    let inner_layers = count_layers("encoder.encoders");
    let tp_layers = count_layers("encoder.tp_encoders");
    let n_layers = inner_layers + 1; // encoders0 + encoders

    let qkv0 = shape("encoder.encoders0.0.self_attn.linear_q_k_v.weight")?;
    let out0 = shape("encoder.encoders0.0.self_attn.linear_out.weight")?;
    let ffn_up = shape("encoder.encoders0.0.feed_forward.w_1.weight")?;
    let fsmn = shape("encoder.encoders0.0.self_attn.fsmn_block.weight")?;
    let ctc_head = shape("ctc.ctc_lo.weight")?;
    let embed = shape("embed.weight")?;

    if qkv0.len() != 2 || out0.len() != 2 || ffn_up.len() != 2 || ctc_head.len() != 2 {
        return Err(validate_error(
            "sensevoice projection tensors must be rank 2".to_string(),
        ));
    }
    let d_model = out0[0] as usize; // linear_out: [d_model, d_model]
    let feature_dim = qkv0[1] as usize; // layer-0 qkv input = LFR feature dim
    let ffn_dim = ffn_up[0] as usize;
    if fsmn.len() != 3 || fsmn[0] as usize != d_model || fsmn[1] != 1 {
        return Err(validate_error(format!(
            "sensevoice fsmn kernel has unexpected shape {fsmn:?} (want [{d_model}, 1, k])"
        )));
    }
    let fsmn_kernel = fsmn[2] as usize;
    if qkv0[0] as usize != 3 * d_model {
        return Err(validate_error(format!(
            "sensevoice qkv output dim {} != 3 * d_model {}",
            qkv0[0],
            3 * d_model
        )));
    }
    if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
        return Err(validate_error(format!(
            "sensevoice attention_heads {n_heads} does not divide d_model {d_model}"
        )));
    }
    let vocab_size = ctc_head[0] as usize;
    if ctc_head[1] as usize != d_model {
        return Err(validate_error(format!(
            "sensevoice ctc head input dim {} != d_model {d_model}",
            ctc_head[1]
        )));
    }
    if vocab_size != vocab_pieces {
        return Err(validate_error(format!(
            "sensevoice ctc head vocab {vocab_size} != spm pieces {vocab_pieces}"
        )));
    }
    if embed.len() != 2 || embed[1] as usize != feature_dim {
        return Err(validate_error(format!(
            "sensevoice prompt embed shape {embed:?} does not end in feature dim {feature_dim}"
        )));
    }

    Ok(SenseVoiceDerivedHparams {
        n_layers,
        tp_layers,
        d_model,
        n_heads,
        ffn_dim,
        fsmn_kernel,
        feature_dim,
        vocab_size,
    })
}

fn build_sensevoice_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: SenseVoiceQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tensor in &safetensors.header().tensors {
        let Some((target_name, force_f32)) = remap_sensevoice_tensor_name(tensor.name.as_str())
        else {
            continue;
        };
        if !seen.insert(target_name.clone()) {
            return Err(validate_error(format!(
                "sensevoice import mapped duplicate destination tensor '{target_name}'"
            )));
        }
        let target_dims = normalize_sensevoice_weight_dims(tensor.shape.as_slice());
        let data = safetensors.tensor_data(tensor)?;
        let tensor_type = quantized_tensor_type_for_sensevoice_tensor(
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
                            "sensevoice quantization failed for '{}' -> '{target_name}' ({qtype:?}): {error}",
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

/// Map a FunASR SenseVoice tensor name to its `.oasr` target name + whether it
/// must be stored f32. Returns `None` to drop a tensor.
fn remap_sensevoice_tensor_name(source_name: &str) -> Option<(String, bool)> {
    if source_name == "ctc.ctc_lo.weight" {
        return Some(("ctc.head.weight".to_string(), false));
    }
    if source_name == "ctc.ctc_lo.bias" {
        return Some(("ctc.head.bias".to_string(), true));
    }
    if source_name == "embed.weight" {
        // 16x560 prompt-embedding table, consumed by get_rows: stays f32.
        return Some(("embed.prompt.weight".to_string(), true));
    }
    if let Some(rest) = source_name.strip_prefix("encoder.after_norm.") {
        return Some((format!("enc.after_norm.{rest}"), true));
    }
    if let Some(rest) = source_name.strip_prefix("encoder.tp_norm.") {
        return Some((format!("tp.norm.{rest}"), true));
    }
    let (scope, rest) = if let Some(rest) = source_name.strip_prefix("encoder.encoders0.") {
        // encoders0.0 -> enc.blk.0
        let (layer, tail) = rest.split_once('.')?;
        if layer != "0" {
            return None;
        }
        ("enc.blk".to_string(), format!("0.{tail}"))
    } else if let Some(rest) = source_name.strip_prefix("encoder.encoders.") {
        // encoders.{i} -> enc.blk.{i+1}
        let (layer, tail) = rest.split_once('.')?;
        let layer: usize = layer.parse().ok()?;
        ("enc.blk".to_string(), format!("{}.{tail}", layer + 1))
    } else if let Some(rest) = source_name.strip_prefix("encoder.tp_encoders.") {
        ("tp.blk".to_string(), rest.to_string())
    } else {
        return None;
    };
    let (layer, tail) = rest.split_once('.')?;
    let suffix = match tail {
        "self_attn.linear_q_k_v.weight" => "attn.qkv.weight",
        "self_attn.linear_q_k_v.bias" => "attn.qkv.bias",
        "self_attn.linear_out.weight" => "attn.out.weight",
        "self_attn.linear_out.bias" => "attn.out.bias",
        "self_attn.fsmn_block.weight" => "attn.fsmn.weight",
        "feed_forward.w_1.weight" => "ffn.up.weight",
        "feed_forward.w_1.bias" => "ffn.up.bias",
        "feed_forward.w_2.weight" => "ffn.down.weight",
        "feed_forward.w_2.bias" => "ffn.down.bias",
        "norm1.weight" => "attn.norm.weight",
        "norm1.bias" => "attn.norm.bias",
        "norm2.weight" => "ffn.norm.weight",
        "norm2.bias" => "ffn.norm.bias",
        _ => return None,
    };
    let target = format!("{scope}.{layer}.{suffix}");
    let force_f32 = sensevoice_tensor_is_f32(&target);
    Some((target, force_f32))
}

/// f32-required tensors: norms, biases, the FSMN depthwise kernels, the CMVN
/// vectors, and the prompt-embedding table (get_rows). Only the 2-D linear
/// projections and the CTC head weight may be quantized.
fn sensevoice_tensor_is_f32(target_name: &str) -> bool {
    target_name.ends_with(".bias")
        || target_name.contains(".norm.")
        || target_name.contains("after_norm")
        || target_name.contains(".fsmn.")
        || target_name.starts_with("frontend.cmvn.")
        || target_name.starts_with("embed.prompt")
}

/// Reverse the dims of rank>=2 weights (torch `[out, in]` -> ggml `[in, out]`
/// for `mul_mat`; the FSMN kernel `[C, 1, K]` -> `[K, 1, C]` for the depthwise
/// im2col path), matching the cohere/parakeet rule. 1-D tensors keep their dims.
fn normalize_sensevoice_weight_dims(source_shape: &[u64]) -> Vec<u64> {
    if source_shape.len() >= 2 {
        let mut dims = source_shape.to_vec();
        dims.reverse();
        dims
    } else {
        source_shape.to_vec()
    }
}

fn quantized_tensor_type_for_sensevoice_tensor(
    name: &str,
    dims: &[u64],
    force_f32: bool,
    quantization: SenseVoiceQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if force_f32 || quantization == SenseVoiceQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return None;
    }
    let ne0 = dims.first().copied()?;
    classify_quant_tensor(ne0, quantization)
}

fn sensevoice_runtime_gguf_metadata(
    hparams: &SenseVoiceDerivedHparams,
    request: &SenseVoiceImportRequest,
    vocab_tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str("general.architecture", SENSEVOICE_GGML_ARCHITECTURE_ID);
    put_str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put_str(OASR_METADATA_KEY_MODEL_FAMILY, SENSEVOICE_MODEL_FAMILY);
    put_str(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        SENSEVOICE_GGML_ARCHITECTURE_ID,
    );
    put_str(
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        SENSEVOICE_AUDIO_FRONTEND_ID,
    );
    put_str(OASR_METADATA_KEY_DECODE_POLICY, SENSEVOICE_DECODE_POLICY_ID);
    put_str(GGML_TOKENIZER_ID_KEY, SENSEVOICE_TOKENIZER_ID);
    put_str("openasr.model.id", &request.model_id);

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32("sensevoice.n_layers", hparams.n_layers as u32);
    put_u32("sensevoice.tp_layers", hparams.tp_layers as u32);
    put_u32("sensevoice.d_model", hparams.d_model as u32);
    put_u32("sensevoice.n_heads", hparams.n_heads as u32);
    put_u32("sensevoice.ffn_dim", hparams.ffn_dim as u32);
    put_u32("sensevoice.fsmn_kernel", hparams.fsmn_kernel as u32);
    put_u32("sensevoice.feature_dim", hparams.feature_dim as u32);
    put_u32("sensevoice.vocab_size", hparams.vocab_size as u32);
    put_u32("ctc.blank_token_id", SENSEVOICE_CTC_BLANK_ID);

    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufWriteValue::StringArray(vocab_tokens.to_vec()),
    );
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaps_layer_zero_and_inner_layers_to_contiguous_blocks() {
        assert_eq!(
            remap_sensevoice_tensor_name("encoder.encoders0.0.self_attn.linear_q_k_v.weight"),
            Some(("enc.blk.0.attn.qkv.weight".to_string(), false))
        );
        assert_eq!(
            remap_sensevoice_tensor_name("encoder.encoders.0.self_attn.linear_out.weight"),
            Some(("enc.blk.1.attn.out.weight".to_string(), false))
        );
        assert_eq!(
            remap_sensevoice_tensor_name("encoder.encoders.48.feed_forward.w_2.weight"),
            Some(("enc.blk.49.ffn.down.weight".to_string(), false))
        );
        assert_eq!(
            remap_sensevoice_tensor_name("encoder.tp_encoders.19.norm1.weight"),
            Some(("tp.blk.19.attn.norm.weight".to_string(), true))
        );
        assert_eq!(
            remap_sensevoice_tensor_name("encoder.encoders0.0.self_attn.fsmn_block.weight"),
            Some(("enc.blk.0.attn.fsmn.weight".to_string(), true))
        );
        assert_eq!(
            remap_sensevoice_tensor_name("ctc.ctc_lo.weight"),
            Some(("ctc.head.weight".to_string(), false))
        );
        assert_eq!(
            remap_sensevoice_tensor_name("embed.weight"),
            Some(("embed.prompt.weight".to_string(), true))
        );
        assert_eq!(
            remap_sensevoice_tensor_name("encoder.after_norm.weight"),
            Some(("enc.after_norm.weight".to_string(), true))
        );
        assert_eq!(remap_sensevoice_tensor_name("some.unknown.tensor"), None);
    }

    #[test]
    fn quantizes_only_eligible_projections() {
        // 512-wide projection: q8_0 under Q8_0, q4_k under Q4_K.
        assert_eq!(
            quantized_tensor_type_for_sensevoice_tensor(
                "enc.blk.1.attn.qkv.weight",
                &[512, 1536],
                false,
                SenseVoiceQuantizationMode::Q8_0,
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        assert_eq!(
            quantized_tensor_type_for_sensevoice_tensor(
                "ctc.head.weight",
                &[512, 25055],
                false,
                SenseVoiceQuantizationMode::Q4_K,
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
        // Layer 0's qkv input is 560 (not 32-aligned): stays f16.
        assert_eq!(
            quantized_tensor_type_for_sensevoice_tensor(
                "enc.blk.0.attn.qkv.weight",
                &[560, 1536],
                false,
                SenseVoiceQuantizationMode::Q8_0,
            ),
            None
        );
        // Norms / fsmn / embed are force_f32 and never quantize.
        assert_eq!(
            quantized_tensor_type_for_sensevoice_tensor(
                "enc.blk.1.attn.fsmn.weight",
                &[11, 1, 512],
                true,
                SenseVoiceQuantizationMode::Q8_0,
            ),
            None
        );
    }

    #[test]
    fn f32_rule_covers_non_matmul_tensors() {
        for name in [
            "enc.blk.0.attn.norm.weight",
            "enc.blk.0.attn.qkv.bias",
            "enc.blk.3.attn.fsmn.weight",
            "enc.after_norm.bias",
            "tp.norm.weight",
            "ctc.head.bias",
            "embed.prompt.weight",
            "frontend.cmvn.neg_mean",
        ] {
            assert!(sensevoice_tensor_is_f32(name), "{name} must stay f32");
        }
        for name in [
            "enc.blk.0.attn.qkv.weight",
            "tp.blk.5.ffn.up.weight",
            "ctc.head.weight",
        ] {
            assert!(!sensevoice_tensor_is_f32(name), "{name} may quantize");
        }
    }

    #[test]
    fn parses_am_mvn_vectors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("am.mvn");
        std::fs::write(
            &path,
            "<Nnet>\n<Splice> 4 4\n[ 0 ]\n<AddShift> 4 4\n<LearnRateCoef> 0 [ -1.0 -2.0 -3.5 -4.25 ]\n<Rescale> 4 4\n<LearnRateCoef> 0 [ 0.5 0.25 2.0 1.0 ]\n</Nnet>\n",
        )
        .expect("write");
        let (neg_mean, inv_stddev) = parse_am_mvn(&path).expect("parse");
        assert_eq!(neg_mean, vec![-1.0, -2.0, -3.5, -4.25]);
        assert_eq!(inv_stddev, vec![0.5, 0.25, 2.0, 1.0]);
    }

    #[test]
    fn am_mvn_rejects_wrong_vector_count() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("am.mvn");
        std::fs::write(&path, "<AddShift> 2 2\n<LearnRateCoef> 0 [ 1 2 ]\n").expect("write");
        assert!(parse_am_mvn(&path).is_err());
    }

    #[test]
    fn parses_minimal_sentencepiece_proto() {
        // Two pieces: "a" (score omitted) and "<s>" with a score field to skip.
        // piece submessage: field1 (tag 0x0a) len str
        let mut proto = Vec::new();
        // pieces[0]: ModelProto field 1 (tag 0x0a), len 3: {0x0a, 1, 'a'}
        proto.extend_from_slice(&[0x0a, 3, 0x0a, 1, b'a']);
        // pieces[1]: {field1 "<s>", field2 float score}
        let sub = [
            0x0a, 3, b'<', b's', b'>', // piece "<s>"
            0x15, 0, 0, 0x80, 0x3f, // field 2 (fixed32 float 1.0)
        ];
        proto.push(0x0a);
        proto.push(sub.len() as u8);
        proto.extend_from_slice(&sub);
        // trailing unrelated field (field 2, varint) must be skipped.
        proto.extend_from_slice(&[0x10, 0x05]);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("spm.model");
        std::fs::write(&path, &proto).expect("write");
        let pieces = parse_sentencepiece_pieces(&path).expect("parse");
        assert_eq!(pieces, vec!["a".to_string(), "<s>".to_string()]);
    }

    #[test]
    fn parses_attention_heads_from_config_yaml() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.yaml");
        std::fs::write(
            &path,
            "encoder: SenseVoiceEncoderSmall\nencoder_conf:\n    output_size: 512\n    attention_heads: 4\n",
        )
        .expect("write");
        assert_eq!(parse_attention_heads(&path).expect("parse"), 4);
    }

    #[test]
    fn metadata_carries_family_and_hparams() {
        let hparams = SenseVoiceDerivedHparams {
            n_layers: 50,
            tp_layers: 20,
            d_model: 512,
            n_heads: 4,
            ffn_dim: 2048,
            fsmn_kernel: 11,
            feature_dim: 560,
            vocab_size: 25055,
        };
        let request = SenseVoiceImportRequest {
            source_root: PathBuf::from("/tmp/src"),
            output_root: PathBuf::from("/tmp/out.oasr"),
            model_id: "sensevoice-small-test".to_string(),
            quantization: SenseVoiceQuantizationMode::Fp16,
        };
        let metadata = sensevoice_runtime_gguf_metadata(&hparams, &request, &["<unk>".to_string()]);
        assert_eq!(
            metadata.get(OASR_METADATA_KEY_MODEL_FAMILY),
            Some(&GgufWriteValue::String(SENSEVOICE_MODEL_FAMILY.to_string()))
        );
        assert_eq!(
            metadata.get("sensevoice.n_layers"),
            Some(&GgufWriteValue::U32(50))
        );
        assert_eq!(
            metadata.get("sensevoice.tp_layers"),
            Some(&GgufWriteValue::U32(20))
        );
        assert_eq!(
            metadata.get("ctc.blank_token_id"),
            Some(&GgufWriteValue::U32(0))
        );
    }
}
