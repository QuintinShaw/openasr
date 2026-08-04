//! Convert a local `FunAudioLLM/Fun-ASR-Nano-2512` FunASR source into an
//! OpenASR `.oasr` (GGUF-v0) runtime pack.
//!
//! The staged source is the two-step shape the other FunASR-derived families
//! already use (see `sensevoice::package_import`): a python prep script
//! (`tooling/publish-model/scripts/funasr_nano_pt_to_safetensors.py`) turns the
//! torch-pickle `model.pt` into a `model.safetensors` keyed by the openasr
//! tensor-name convention plus a `funasr_nano_meta.json` architecture sidecar,
//! and this importer then validates, quantizes, and packs through the shared
//! shared GGUF writer -- the same path every other family uses, so the
//! `.oasr` v1 required-metadata contract (`openasr.package.version = "1"` and
//! the descriptor keys) is stamped by construction rather than by a one-off
//! pack script remembering every key.
//!
//! Tensor naming: the SAN-M encoder half reuses the `sensevoice` convention
//! (`enc.blk.{i}.*` / `tp.blk.{i}.*` / `enc.after_norm.*` / `tp.norm.*`), the
//! adaptor keeps its own `adaptor.*` scope, and the stock Qwen3-0.6B decoder
//! uses `blk.{i}.*` / `token_embd` / `output` / `output_norm` (see
//! `tensor_names`). Dims are written ggml-style: rank>=2 safetensors shapes
//! (torch `[out, in]` row-major) are relabeled to `[in, out]` with the flat
//! byte layout kept, matching every other importer in this crate; the FSMN
//! depthwise kernel `[D, 1, K]` becomes `[K, 1, D]` for the encoder graph's
//! `reshape_4d(K, 1, 1, D)` read.
//!
//! Quantization follows the shared policy (`models::pack_quant`): FSMN kernels,
//! norms, biases, and every other non-rank-2 tensor stay F32; unaligned rank-2
//! weights stay F16; the SAN-M/tp encoder + adaptor half carries the Q8_0
//! audio-encoder floor; the Qwen3 decoder half takes the full requested rung.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::Deserialize;

use crate::VerifiedPack;
use crate::arch::FUNASR_NANO_GGML_ARCHITECTURE_ID;
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
};
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f16_bits,
    decode_safetensors_payload_as_f32, encode_f16_bits_le, load_gpt2_bpe_merges,
    load_gpt2_bpe_vocab_tokens, pad_gpt2_bpe_vocab_tokens, read_source_json_file, validate_error,
    validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OasrPackWriter, PackEnvelope, TOKENIZER_GGML_MERGES_KEY, TOKENIZER_GGML_MODEL_KEY,
    TOKENIZER_GGML_TOKENS_KEY,
};
use crate::models::pack_quant::{
    PackQuant, QuantizedAxis, TensorQuantizationContract, TensorRole, classify_quant_tensor_role,
};

use super::runtime_contract::{
    FUNASR_NANO_ENCODER_LAYER_NORM_EPSILON, FUNASR_NANO_RMS_NORM_EPSILON, FUNASR_NANO_ROPE_THETA,
};
use super::tensor_names::AUDIO_ENCODER_TENSOR_NAME_PREFIXES;

pub(crate) const TENSOR_QUANTIZATION_CONTRACT: TensorQuantizationContract =
    TensorQuantizationContract::SemanticRolesV1 {
        model_architecture: FUNASR_NANO_GGML_ARCHITECTURE_ID,
        classify: classify_funasr_nano_quant_tensor_role,
        quantized_axis: QuantizedAxis::First,
    };

fn classify_funasr_nano_quant_tensor_role(name: &str) -> TensorRole {
    if name.ends_with(".weight")
        && AUDIO_ENCODER_TENSOR_NAME_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
    {
        TensorRole::AcousticEncoderMatrix
    } else if name.ends_with(".weight") {
        TensorRole::TextDecoderMatrix
    } else {
        TensorRole::NonQuantizable
    }
}

const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";
const SOURCE_META_JSON: &str = "funasr_nano_meta.json";
/// The stock Qwen3-0.6B tokenizer source directory staged alongside the prep
/// outputs (vocab.json + merges.txt + tokenizer_config.json).
const SOURCE_QWEN3_DIR: &str = "Qwen3-0.6B";
const TOKENIZER_GGML_MODEL_VALUE_GPT2: &str = "gpt2";
const OPENASR_MODEL_ID_KEY: &str = "openasr.model.id";

pub type FunasrNanoQuantizationMode = PackQuant;

#[derive(Debug, Clone)]
pub struct FunasrNanoImportRequest {
    pub source_root: PathBuf,
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: FunasrNanoQuantizationMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunasrNanoImportResult {
    pub output_path: PathBuf,
    pub verified_pack: VerifiedPack,
    pub tensor_count: usize,
    pub vocab_size: usize,
}

/// Architecture scalars emitted by the prep script's `--out_meta` sidecar. The
/// importer treats the safetensors shapes as ground truth and cross-checks
/// every derivable value against this sidecar (fail-closed on mismatch), so a
/// stale or hand-edited sidecar cannot silently produce a mislabeled pack.
#[derive(Debug, Clone, Deserialize)]
struct FunasrNanoMetaJson {
    enc: FunasrNanoEncMeta,
    adp: FunasrNanoAdpMeta,
    llm: FunasrNanoLlmMeta,
}

#[derive(Debug, Clone, Deserialize)]
struct FunasrNanoEncMeta {
    n_layers: usize,
    tp_blocks: usize,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    fsmn_kernel: usize,
    feature_dim: usize,
    #[serde(default)]
    layer_norm_eps: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct FunasrNanoAdpMeta {
    n_layers: usize,
    n_heads: usize,
    llm_dim: usize,
    encoder_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct FunasrNanoLlmMeta {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_positions: usize,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    rms_norm_eps: Option<f64>,
    chatml_im_start_token_id: u32,
    chatml_im_end_token_id: u32,
    endoftext_token_id: u32,
}

pub fn convert_local_funasr_nano_source_to_runtime_pack(
    request: &FunasrNanoImportRequest,
) -> Result<FunasrNanoImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_root)?;
    if request.model_id.trim().is_empty() {
        return Err(validate_error(
            "funasr-nano local-source converter requires non-empty model_id",
        ));
    }
    let meta: FunasrNanoMetaJson = read_source_json_file(&request.source_root, SOURCE_META_JSON)?;
    let safetensors = SafetensorsFile::open(request.source_root.join(SOURCE_MODEL_SAFETENSORS))?;
    validate_funasr_nano_source(&safetensors, &meta)?;

    // Stock Qwen3-0.6B GPT-2 BPE tokenizer, staged under the source root by
    // the publish pipeline's download/prep stages. The token array must be
    // exactly the declared embedding vocab wide (the runtime tokenizer fails
    // closed on any other length), so pad with the shared placeholder scheme.
    let qwen_root = request.source_root.join(SOURCE_QWEN3_DIR);
    let mut tokens = load_gpt2_bpe_vocab_tokens(&qwen_root, "funasr-nano")?;
    let merges = load_gpt2_bpe_merges(&qwen_root, "funasr-nano")?;
    pad_gpt2_bpe_vocab_tokens(&mut tokens, meta.llm.vocab_size, "funasr-nano")?;
    validate_chatml_token_ids(&meta, &tokens)?;

    let tensors = build_funasr_nano_runtime_tensors(&safetensors, request.quantization)?;
    let metadata = funasr_nano_runtime_gguf_metadata(&meta, request, &tokens, &merges);

    let verified = OasrPackWriter::write(
        &request.output_root,
        PackEnvelope::asr(FUNASR_NANO_GGML_ARCHITECTURE_ID),
        metadata,
        &tensors,
    )
    .map_err(|error| {
        validate_error(format!(
            "funasr-nano OASR writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let tensor_count = verified.preflight().tensor_index().tensors().len();
    Ok(FunasrNanoImportResult {
        output_path: request.output_root.clone(),
        verified_pack: verified,
        tensor_count,
        vocab_size: meta.llm.vocab_size,
    })
}

/// Fail-closed cross-check between the prep sidecar's architecture scalars and
/// the actual safetensors inventory: the exact expected tensor-name set must be
/// present (nothing missing, nothing extra), and every shape the runtime's
/// graphs depend on must agree with the sidecar's declared scalars. A stale or
/// hand-edited sidecar therefore rejects the import instead of writing a
/// mislabeled pack.
fn validate_funasr_nano_source(
    safetensors: &SafetensorsFile,
    meta: &FunasrNanoMetaJson,
) -> Result<(), LocalSourceImportError> {
    let enc = &meta.enc;
    let adp = &meta.adp;
    let llm = &meta.llm;
    for (label, value) in [
        ("enc.n_layers", enc.n_layers),
        ("enc.tp_blocks", enc.tp_blocks),
        ("enc.d_model", enc.d_model),
        ("enc.n_heads", enc.n_heads),
        ("enc.head_dim", enc.head_dim),
        ("enc.ffn_dim", enc.ffn_dim),
        ("enc.fsmn_kernel", enc.fsmn_kernel),
        ("enc.feature_dim", enc.feature_dim),
        ("adp.n_layers", adp.n_layers),
        ("adp.n_heads", adp.n_heads),
        ("adp.llm_dim", adp.llm_dim),
        ("adp.encoder_dim", adp.encoder_dim),
        ("llm.n_layers", llm.n_layers),
        ("llm.d_model", llm.d_model),
        ("llm.n_heads", llm.n_heads),
        ("llm.n_kv_heads", llm.n_kv_heads),
        ("llm.head_dim", llm.head_dim),
        ("llm.ffn_dim", llm.ffn_dim),
        ("llm.vocab_size", llm.vocab_size),
        ("llm.max_positions", llm.max_positions),
    ] {
        if value == 0 {
            return Err(validate_error(format!(
                "funasr-nano meta '{label}' must be positive"
            )));
        }
    }
    if enc.n_heads * enc.head_dim != enc.d_model {
        return Err(validate_error(format!(
            "funasr-nano enc n_heads {} * head_dim {} != d_model {}",
            enc.n_heads, enc.head_dim, enc.d_model
        )));
    }
    if enc.fsmn_kernel.is_multiple_of(2) {
        return Err(validate_error(format!(
            "funasr-nano enc fsmn_kernel {} must be odd (mirrored padding)",
            enc.fsmn_kernel
        )));
    }
    if !llm.n_heads.is_multiple_of(llm.n_kv_heads) {
        return Err(validate_error(format!(
            "funasr-nano llm n_heads {} is not divisible by n_kv_heads {}",
            llm.n_heads, llm.n_kv_heads
        )));
    }
    // The norm epsilons and rope theta are family constants on the runtime
    // side (the graphs never read them from pack metadata); a sidecar that
    // declares different values describes a different checkpoint shape, so
    // the import fails closed instead of silently packing a mismatch.
    if let Some(eps) = enc.layer_norm_eps {
        check_sidecar_float_near(
            eps,
            FUNASR_NANO_ENCODER_LAYER_NORM_EPSILON,
            "enc.layer_norm_eps",
        )?;
    }
    if let Some(theta) = llm.rope_theta {
        check_sidecar_float_near(theta, FUNASR_NANO_ROPE_THETA, "llm.rope_theta")?;
    }
    if let Some(eps) = llm.rms_norm_eps {
        check_sidecar_float_near(eps, FUNASR_NANO_RMS_NORM_EPSILON, "llm.rms_norm_eps")?;
    }
    // The adaptor bridges the two halves; a sidecar that disagrees with the
    // tensor shapes about either width would build an unbindable graph.
    if adp.encoder_dim != enc.d_model {
        return Err(validate_error(format!(
            "funasr-nano adp.encoder_dim {} != enc.d_model {}",
            adp.encoder_dim, enc.d_model
        )));
    }
    if adp.llm_dim != llm.d_model {
        return Err(validate_error(format!(
            "funasr-nano adp.llm_dim {} != llm.d_model {}",
            adp.llm_dim, llm.d_model
        )));
    }

    let shape_by_name: BTreeMap<&str, &[u64]> = safetensors
        .header()
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
        .collect();
    if shape_by_name.len() != safetensors.header().tensors.len() {
        return Err(validate_error(
            "funasr-nano source safetensors carries duplicate tensor names",
        ));
    }

    // Two adaptor intermediates are not declared in the sidecar (they are
    // checkpoint facts, not runtime hparams): derive them from the shapes and
    // cross-check the paired tensor that must agree.
    let adp_intermediate = require_shape(&shape_by_name, "adaptor.linear1.weight")?[0];
    if require_shape(&shape_by_name, "adaptor.linear2.weight")?[1] != adp_intermediate {
        return Err(validate_error(format!(
            "funasr-nano adaptor.linear2 input dim disagrees with adaptor.linear1 output dim {adp_intermediate}"
        )));
    }
    let adp_ffn_intermediate = require_shape(&shape_by_name, "adaptor.blk.0.ffn.up.weight")?[0];

    let expected = expected_tensor_inventory(meta, adp_intermediate, adp_ffn_intermediate);
    for (name, shape) in &expected {
        let actual = shape_by_name.get(name.as_str()).ok_or_else(|| {
            validate_error(format!("funasr-nano source is missing tensor '{name}'"))
        })?;
        if *actual != shape.as_slice() {
            return Err(validate_error(format!(
                "funasr-nano tensor '{name}' has shape {actual:?}, expected {shape:?}"
            )));
        }
    }
    for tensor in &safetensors.header().tensors {
        if !expected.contains_key(&tensor.name) {
            return Err(validate_error(format!(
                "funasr-nano source carries unexpected tensor '{}'",
                tensor.name
            )));
        }
    }
    Ok(())
}

/// Fail closed when an optional sidecar scalar disagrees with the runtime's
/// family constant beyond float-representation noise (the sidecar is f64 JSON,
/// the constants are f32, so an exact comparison would reject the identical
/// value).
fn check_sidecar_float_near(
    sidecar_value: f64,
    runtime_constant: f32,
    label: &str,
) -> Result<(), LocalSourceImportError> {
    let expected = runtime_constant as f64;
    let tolerance = expected.abs().max(1.0) * 1e-6;
    if (sidecar_value - expected).abs() > tolerance {
        return Err(validate_error(format!(
            "funasr-nano meta '{label}' {sidecar_value} disagrees with the runtime constant {expected}"
        )));
    }
    Ok(())
}

fn require_shape<'a>(
    shape_by_name: &BTreeMap<&str, &'a [u64]>,
    name: &str,
) -> Result<&'a [u64], LocalSourceImportError> {
    shape_by_name
        .get(name)
        .copied()
        .ok_or_else(|| validate_error(format!("funasr-nano source is missing tensor '{name}'")))
}

/// The complete `.oasr` tensor inventory with expected torch-layout (`[out,
/// in]`) shapes, generated from the validated sidecar scalars. Serves both as
/// the import-time shape oracle and as unit-testable documentation of the
/// pack layout.
fn expected_tensor_inventory(
    meta: &FunasrNanoMetaJson,
    adp_intermediate: u64,
    adp_ffn_intermediate: u64,
) -> BTreeMap<String, Vec<u64>> {
    let enc = &meta.enc;
    let adp = &meta.adp;
    let llm = &meta.llm;
    let d = enc.d_model as u64;
    let feature = enc.feature_dim as u64;
    let ffn = enc.ffn_dim as u64;
    let kernel = enc.fsmn_kernel as u64;
    let llm_d = llm.d_model as u64;
    let llm_ffn = llm.ffn_dim as u64;
    let q_out = (llm.n_heads * llm.head_dim) as u64;
    let kv_out = (llm.n_kv_heads * llm.head_dim) as u64;
    let vocab = llm.vocab_size as u64;
    let adp_dim = adp.llm_dim as u64;

    let mut out: BTreeMap<String, Vec<u64>> = BTreeMap::new();

    // SAN-M blocks: layer 0's input norm and fused QKV read the raw LFR
    // feature width; every later block reads the encoder model width.
    sanm_block_shapes(&mut out, "enc.blk.0", d, ffn, kernel, feature);
    for i in 1..enc.n_layers {
        sanm_block_shapes(&mut out, &format!("enc.blk.{i}"), d, ffn, kernel, d);
    }
    out.insert("enc.after_norm.weight".to_string(), vec![d]);
    out.insert("enc.after_norm.bias".to_string(), vec![d]);
    for j in 0..enc.tp_blocks {
        sanm_block_shapes(&mut out, &format!("tp.blk.{j}"), d, ffn, kernel, d);
    }
    out.insert("tp.norm.weight".to_string(), vec![d]);
    out.insert("tp.norm.bias".to_string(), vec![d]);

    // Adaptor: encoder-width -> intermediate -> LLM-width MLP bridge plus
    // standard transformer blocks over the LLM width.
    out.insert(
        "adaptor.linear1.weight".to_string(),
        vec![adp_intermediate, enc.d_model as u64],
    );
    out.insert("adaptor.linear1.bias".to_string(), vec![adp_intermediate]);
    out.insert(
        "adaptor.linear2.weight".to_string(),
        vec![adp_dim, adp_intermediate],
    );
    out.insert("adaptor.linear2.bias".to_string(), vec![adp_dim]);
    for i in 0..adp.n_layers {
        let prefix = format!("adaptor.blk.{i}");
        out.insert(format!("{prefix}.attn.norm.weight"), vec![adp_dim]);
        out.insert(format!("{prefix}.attn.norm.bias"), vec![adp_dim]);
        for proj in ["q", "k", "v"] {
            out.insert(
                format!("{prefix}.attn.{proj}.weight"),
                vec![adp_dim, adp_dim],
            );
            out.insert(format!("{prefix}.attn.{proj}.bias"), vec![adp_dim]);
        }
        out.insert(format!("{prefix}.attn.out.weight"), vec![adp_dim, adp_dim]);
        out.insert(format!("{prefix}.attn.out.bias"), vec![adp_dim]);
        out.insert(format!("{prefix}.ffn.norm.weight"), vec![adp_dim]);
        out.insert(format!("{prefix}.ffn.norm.bias"), vec![adp_dim]);
        out.insert(
            format!("{prefix}.ffn.up.weight"),
            vec![adp_ffn_intermediate, adp_dim],
        );
        out.insert(format!("{prefix}.ffn.up.bias"), vec![adp_ffn_intermediate]);
        out.insert(
            format!("{prefix}.ffn.down.weight"),
            vec![adp_dim, adp_ffn_intermediate],
        );
        out.insert(format!("{prefix}.ffn.down.bias"), vec![adp_dim]);
    }

    // Stock Qwen3-0.6B decoder under the bare `blk.N` scope.
    out.insert("token_embd.weight".to_string(), vec![vocab, llm_d]);
    out.insert("output.weight".to_string(), vec![vocab, llm_d]);
    out.insert("output_norm.weight".to_string(), vec![llm_d]);
    for i in 0..llm.n_layers {
        let prefix = format!("blk.{i}");
        out.insert(format!("{prefix}.attn_norm.weight"), vec![llm_d]);
        out.insert(format!("{prefix}.attn_q.weight"), vec![q_out, llm_d]);
        out.insert(format!("{prefix}.attn_k.weight"), vec![kv_out, llm_d]);
        out.insert(format!("{prefix}.attn_v.weight"), vec![kv_out, llm_d]);
        out.insert(format!("{prefix}.attn_output.weight"), vec![llm_d, q_out]);
        out.insert(
            format!("{prefix}.attn_q_norm.weight"),
            vec![llm.head_dim as u64],
        );
        out.insert(
            format!("{prefix}.attn_k_norm.weight"),
            vec![llm.head_dim as u64],
        );
        out.insert(format!("{prefix}.ffn_norm.weight"), vec![llm_d]);
        out.insert(format!("{prefix}.ffn_gate.weight"), vec![llm_ffn, llm_d]);
        out.insert(format!("{prefix}.ffn_up.weight"), vec![llm_ffn, llm_d]);
        out.insert(format!("{prefix}.ffn_down.weight"), vec![llm_d, llm_ffn]);
    }
    out
}

/// The 13-tensor SAN-M block inventory (see `sensevoice::package_import`'s
/// identical per-layer layout): fused QKV + gated FSMN depthwise attention and
/// a two-layer FFN, norms and biases included. `input_dim` is the block's
/// input width: the model width for every block except `enc.blk.0`, whose
/// input norm + QKV read the raw LFR feature width.
fn sanm_block_shapes(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    d: u64,
    ffn: u64,
    kernel: u64,
    input_dim: u64,
) {
    out.insert(format!("{prefix}.attn.norm.weight"), vec![input_dim]);
    out.insert(format!("{prefix}.attn.norm.bias"), vec![input_dim]);
    out.insert(format!("{prefix}.attn.qkv.weight"), vec![3 * d, input_dim]);
    out.insert(format!("{prefix}.attn.qkv.bias"), vec![3 * d]);
    out.insert(format!("{prefix}.attn.out.weight"), vec![d, d]);
    out.insert(format!("{prefix}.attn.out.bias"), vec![d]);
    out.insert(format!("{prefix}.attn.fsmn.weight"), vec![d, 1, kernel]);
    out.insert(format!("{prefix}.ffn.norm.weight"), vec![d]);
    out.insert(format!("{prefix}.ffn.norm.bias"), vec![d]);
    out.insert(format!("{prefix}.ffn.up.weight"), vec![ffn, d]);
    out.insert(format!("{prefix}.ffn.up.bias"), vec![ffn]);
    out.insert(format!("{prefix}.ffn.down.weight"), vec![d, ffn]);
    out.insert(format!("{prefix}.ffn.down.bias"), vec![d]);
}

/// The three ChatML control token ids are decode-time facts the executor
/// trusts blindly. Cross-check that each id the prep sidecar declares lands
/// inside the packed vocab and points at a real, non-placeholder token, so a
/// sidecar/source skew fails the import instead of decoding garbage.
fn validate_chatml_token_ids(
    meta: &FunasrNanoMetaJson,
    tokens: &[String],
) -> Result<(), LocalSourceImportError> {
    let llm = &meta.llm;
    for (label, id) in [
        ("chatml_im_start_token_id", llm.chatml_im_start_token_id),
        ("chatml_im_end_token_id", llm.chatml_im_end_token_id),
        ("endoftext_token_id", llm.endoftext_token_id),
    ] {
        let token = tokens.get(id as usize).ok_or_else(|| {
            validate_error(format!(
                "funasr-nano meta {label} {id} is outside the declared vocab size {}",
                llm.vocab_size
            ))
        })?;
        if token.is_empty() || token.starts_with("<unused_") {
            return Err(validate_error(format!(
                "funasr-nano meta {label} {id} points at an unused placeholder token"
            )));
        }
    }
    Ok(())
}

fn build_funasr_nano_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: FunasrNanoQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::with_capacity(safetensors.header().tensors.len());
    for tensor in &safetensors.header().tensors {
        // The source inventory is fail-closed validated before this runs, so
        // every tensor here is a known runtime tensor.
        let dims = ggml_dims_for_pack(tensor.shape.as_slice());
        let storage = funasr_nano_tensor_storage(tensor.name.as_str(), &dims, quantization);
        let data = safetensors.tensor_data(tensor)?;
        let write_tensor = match storage {
            FunasrNanoTensorStorage::F32 => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                GgufWriteTensor {
                    name: tensor.name.clone(),
                    dims,
                    tensor_type: GgufWriteTensorType::F32,
                    data: bytes,
                }
            }
            FunasrNanoTensorStorage::F16 => {
                let bits =
                    decode_safetensors_payload_as_f16_bits(&tensor.name, &tensor.dtype, data)?;
                GgufWriteTensor {
                    name: tensor.name.clone(),
                    dims,
                    tensor_type: GgufWriteTensorType::F16,
                    data: encode_f16_bits_le(bits),
                }
            }
            FunasrNanoTensorStorage::Quantized(qtype) => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let quantized =
                    quantize_f32_to_ggml_tensor_data(qtype, &dims, &values).map_err(|error| {
                        validate_error(format!(
                            "funasr-nano quantization failed for '{}' ({qtype:?}): {error}",
                            tensor.name
                        ))
                    })?;
                GgufWriteTensor {
                    name: tensor.name.clone(),
                    dims,
                    tensor_type: qtype,
                    data: quantized,
                }
            }
        };
        out.push(write_tensor);
    }
    Ok(out)
}

/// Safetensors carries torch `[out, in]` row-major bytes, which are already
/// the correct flat layout for ggml `mul_mat`; the converter only relabels the
/// two extents to ggml `[in, out]`. Rank-3+ tensors (the FSMN depthwise kernel
/// `[D, 1, K]` -> `[K, 1, D]`) reverse their extents the same way, matching
/// the encoder graph's `reshape_4d(K, 1, 1, D)` read. 1-D tensors keep theirs.
fn ggml_dims_for_pack(shape: &[u64]) -> Vec<u64> {
    if shape.len() >= 2 {
        let mut dims = shape.to_vec();
        dims.reverse();
        dims
    } else {
        shape.to_vec()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FunasrNanoTensorStorage {
    F32,
    F16,
    Quantized(GgufWriteTensorType),
}

/// f32-required tensors: norms, biases, the FSMN depthwise kernels, and every
/// other non-rank-2 operand (none of them is a `mul_mat` weight). Only rank-2
/// `.weight` matrices are quantization candidates.
fn funasr_nano_tensor_is_f32(name: &str, rank: usize) -> bool {
    rank != 2 || name.contains("norm") || name.ends_with(".bias") || name.contains(".fsmn.")
}

fn funasr_nano_tensor_storage(
    name: &str,
    dims: &[u64],
    quantization: FunasrNanoQuantizationMode,
) -> FunasrNanoTensorStorage {
    if funasr_nano_tensor_is_f32(name, dims.len()) {
        return FunasrNanoTensorStorage::F32;
    }
    if quantization == FunasrNanoQuantizationMode::Fp16 {
        return FunasrNanoTensorStorage::F16;
    }
    if !name.ends_with(".weight") || dims.len() != 2 {
        return FunasrNanoTensorStorage::F16;
    }
    match classify_quant_tensor_role(
        dims,
        quantization,
        classify_funasr_nano_quant_tensor_role(name),
        QuantizedAxis::First,
    ) {
        Some(qtype) => FunasrNanoTensorStorage::Quantized(qtype),
        // Unaligned matmul widths keep the (higher-precision) fp16-mode
        // representation, same fallback every other importer uses.
        None => FunasrNanoTensorStorage::F16,
    }
}

fn funasr_nano_runtime_gguf_metadata(
    meta: &FunasrNanoMetaJson,
    request: &FunasrNanoImportRequest,
    vocab_tokens: &[String],
    merges: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let enc = &meta.enc;
    let adp = &meta.adp;
    let llm = &meta.llm;
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };

    put_str(OPENASR_MODEL_ID_KEY, &request.model_id);
    put_str(TOKENIZER_GGML_MODEL_KEY, TOKENIZER_GGML_MODEL_VALUE_GPT2);

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };

    put_u32("funasr.enc.n_layers", enc.n_layers as u32);
    put_u32("funasr.enc.tp_blocks", enc.tp_blocks as u32);
    put_u32("funasr.enc.d_model", enc.d_model as u32);
    put_u32("funasr.enc.n_heads", enc.n_heads as u32);
    put_u32("funasr.enc.head_dim", enc.head_dim as u32);
    put_u32("funasr.enc.ffn_dim", enc.ffn_dim as u32);
    put_u32("funasr.enc.fsmn_kernel", enc.fsmn_kernel as u32);
    put_u32("funasr.enc.feature_dim", enc.feature_dim as u32);
    put_u32("funasr.adp.n_layers", adp.n_layers as u32);
    put_u32("funasr.adp.n_heads", adp.n_heads as u32);
    put_u32("funasr.adp.encoder_dim", adp.encoder_dim as u32);
    put_u32("funasr.adp.llm_dim", adp.llm_dim as u32);
    put_u32("funasr.llm.n_layers", llm.n_layers as u32);
    put_u32("funasr.llm.d_model", llm.d_model as u32);
    put_u32("funasr.llm.n_heads", llm.n_heads as u32);
    put_u32("funasr.llm.n_kv_heads", llm.n_kv_heads as u32);
    put_u32("funasr.llm.head_dim", llm.head_dim as u32);
    put_u32("funasr.llm.ffn_dim", llm.ffn_dim as u32);
    put_u32("funasr.llm.vocab_size", llm.vocab_size as u32);
    put_u32("funasr.llm.max_positions", llm.max_positions as u32);
    put_u32(
        "funasr.llm.chatml_im_start_token_id",
        llm.chatml_im_start_token_id,
    );
    put_u32(
        "funasr.llm.chatml_im_end_token_id",
        llm.chatml_im_end_token_id,
    );
    put_u32("funasr.llm.endoftext_token_id", llm.endoftext_token_id);

    metadata.insert(
        TOKENIZER_GGML_TOKENS_KEY.to_string(),
        GgufWriteValue::StringArray(vocab_tokens.to_vec()),
    );
    metadata.insert(
        TOKENIZER_GGML_MERGES_KEY.to_string(),
        GgufWriteValue::StringArray(merges.to_vec()),
    );

    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_meta() -> FunasrNanoMetaJson {
        FunasrNanoMetaJson {
            enc: FunasrNanoEncMeta {
                n_layers: 50,
                tp_blocks: 20,
                d_model: 512,
                n_heads: 4,
                head_dim: 128,
                ffn_dim: 2048,
                fsmn_kernel: 11,
                feature_dim: 560,
                layer_norm_eps: Some(1e-5),
            },
            adp: FunasrNanoAdpMeta {
                n_layers: 2,
                n_heads: 8,
                llm_dim: 1024,
                encoder_dim: 512,
            },
            llm: FunasrNanoLlmMeta {
                n_layers: 28,
                d_model: 1024,
                n_heads: 16,
                n_kv_heads: 8,
                head_dim: 128,
                ffn_dim: 3072,
                vocab_size: 151_936,
                max_positions: 40_960,
                rope_theta: Some(1_000_000.0),
                rms_norm_eps: Some(1e-6),
                chatml_im_start_token_id: 151_644,
                chatml_im_end_token_id: 151_645,
                endoftext_token_id: 151_643,
            },
        }
    }

    fn fixture_request() -> FunasrNanoImportRequest {
        FunasrNanoImportRequest {
            source_root: PathBuf::from("/tmp/funasr-nano-src"),
            output_root: PathBuf::from("/tmp/funasr-nano.oasr"),
            model_id: "funasr-nano".to_string(),
            quantization: FunasrNanoQuantizationMode::Fp16,
        }
    }

    fn string_metadata(metadata: &BTreeMap<String, GgufWriteValue>, key: &str) -> Option<String> {
        match metadata.get(key)? {
            GgufWriteValue::String(value) => Some(value.clone()),
            _ => None,
        }
    }

    /// Family metadata contains only family-owned keys. The shared writer owns
    /// every package/routing key, so this importer cannot recreate the FunASR
    /// missing-version failure by omission or override.
    #[test]
    fn metadata_does_not_redeclare_envelope_owned_keys() {
        let metadata = funasr_nano_runtime_gguf_metadata(
            &fixture_meta(),
            &fixture_request(),
            &["a".to_string(), "b".to_string()],
            &["a b".to_string()],
        );

        for key in [
            "general.architecture",
            "openasr.package.version",
            "openasr.model.family",
            "openasr.model.architecture",
            "openasr.audio.frontend",
            "openasr.decode.policy",
            "ggml.tokenizer.id",
        ] {
            assert!(!metadata.contains_key(key), "envelope owns {key}");
        }
        assert_eq!(
            string_metadata(&metadata, OPENASR_MODEL_ID_KEY),
            Some("funasr-nano".to_string())
        );
    }
}
