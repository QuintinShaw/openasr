//! Convert a local Dolphin WeNet checkpoint (exported `full.safetensors` +
//! `units.txt` char vocab, with `global_cmvn` folded into the encoder tensors)
//! into an OpenASR `.oasr` (GGUF-v0) runtime pack at fp16, q8_0, or q4_k.
//!
//! Naming contract: the encoder/decoder/CTC tensors are stored under their
//! **exact WeNet state-dict names**. At fp16 they keep raw element order (only
//! f32 -> f16 for the rank>=2 weight matrices/convs; the 1-D biases/norms and the
//! CMVN vectors stay f32). This keeps the runtime executor trivial and, crucially,
//! lets it feed the already parity-verified `encoder_graph::encode()` byte-for-byte
//! the same tensor buffers the raw-safetensors parity harness used — the only delta
//! is the fp16 rounding of the weights.
//!
//! Quantization (q8_0/q4_k) targets only the rank-2 `.weight` projection/embed/
//! output matrices. ggml block-quantizes along ne0 (the contiguous axis), so those
//! matrices are stored with their dims **reversed** (`[out, in]` -> `[in, out]`)
//! exactly as the xasr/cohere importers do: that puts the row-major-innermost `in`
//! dimension on ne0, block-aligns it (`in % 32`/`in % 256`), and groups each row's
//! input features into a superblock. The runtime never trusts the stored dims for
//! layout — it re-declares each graph tensor and only consumes the dequantized f32
//! by element count — so a reversed-dim q-tensor dequantizes to the *same* raw
//! element order an fp16 tensor would (round-trip is order-preserving); the only
//! delta is the quantization rounding. Convs (rank>2), position tables, the CMVN
//! vectors, 1-D biases/norms, and the mel filterbank are never quantized (their
//! ne0 is either tiny/unaligned or numerically sensitive) and keep their fp16-mode
//! representation.
//!
//! Baked into the pack:
//!   * every `encoder.*` / `decoder.*` / `ctc.*` tensor,
//!   * the `context_module.*` native deep-biasing hotword tensors (the
//!     `context_extractor` BiLSTM + word embedding, `context_encoder`,
//!     `biasing_layer` cross-attention, `combiner`, `norm_aft_combiner`) --
//!     everything the inference-time fusion in
//!     `models::dolphin::hotword_context` needs. Only the training-only aux CTC
//!     head over hotword context (`context_module.context_decoder*`) is dropped:
//!     it is never exercised at inference (see
//!     `models::dolphin::hotword_context` for the upstream reference),
//!   * the global CMVN mean/istd (already present as `encoder.global_cmvn.*`),
//!   * a kaldi/HTK mel filterbank (`dolphin.mel_filters`) reconstructed from the
//!     `train.yaml` fbank config for the later frontend phase,
//!   * the char tokenizer (`tokenizer.ggml.tokens`, ids in `units.txt` order) —
//!     the full special-token block, so every advertised `<REGION>` dialect code
//!     resolves at request time (no single region baked as the default; the
//!     importer asserts each advertised code's prefix builds against the vocab),
//!   * the runtime scalar contract keys the install gate validates.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::arch::{
    DOLPHIN_AUDIO_FRONTEND_ID, DOLPHIN_DECODE_POLICY_ID, DOLPHIN_GGML_ARCHITECTURE_ID,
    DOLPHIN_MODEL_FAMILY, DOLPHIN_TOKENIZER_ID,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    read_gguf_tensor_index, write_gguf_file_v0,
};
use crate::models::dolphin::language::{
    DOLPHIN_DEFAULT_LANGUAGE_CODE, build_dolphin_decode_prefix,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
use crate::models::language::REGISTERED_DIALECT_CODES;
use crate::models::local_source_import::{
    LocalSourceImportError, SafetensorsFile, decode_safetensors_payload_as_f32, encode_f16_bits_le,
    f32_to_f16_bits, validate_error, validate_output_pack_extension,
};
use crate::models::oasr_metadata::{
    OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_DECODE_POLICY,
    OASR_METADATA_KEY_MODEL_ARCHITECTURE, OASR_METADATA_KEY_MODEL_FAMILY,
    OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1,
};

// --- fixed small.cn E-Branchformer / Transformer configuration ----------------
// Cross-checked against the checkpoint below (layer counts, d_model, vocab, mel
// dim) so a mismatched export fails closed rather than mislabels the pack.
const ENCODER_N_LAYERS: usize = 12;
const ENCODER_D_MODEL: usize = 768;
const ENCODER_N_HEADS: usize = 12;
const ENCODER_HEAD_DIM: usize = 64;
const ENCODER_FFN_DIM: usize = 3072;
const ENCODER_CGMLP_UNITS: usize = 3072;
const ENCODER_CGMLP_KERNEL: usize = 31;
const ENCODER_MERGE_KERNEL: usize = 31;
const DECODER_N_LAYERS: usize = 12;
const DECODER_N_HEADS: usize = 12;
const DECODER_FFN_DIM: usize = 3072;
const DECODER_MAX_CTX: usize = 5000;
const FEATURE_DIM: usize = 80;
const SOS_TOKEN_ID: u32 = 2;
const EOS_TOKEN_ID: u32 = 3;
const CTC_BLANK_TOKEN_ID: u32 = 0;

// fbank config from `train.yaml` (`fbank_conf`): 25 ms window, 10 ms shift, 80
// mel bins, 16 kHz. Kaldi rounds the 400-sample window up to the next power of
// two for the FFT.
const SAMPLE_RATE_HZ: u32 = 16_000;
const FRAME_LENGTH_MS: u32 = 25;
const FRAME_SHIFT_MS: u32 = 10;
const FFT_SIZE: usize = 512;
const MEL_LOW_HZ: f32 = 20.0;

/// WeNet state-dict namespaces baked into the runtime pack (in order). The
/// hotword deep-biasing module (`context_module.*`) is included here too --
/// only its training-only aux CTC head (`DROPPED_TENSOR_PREFIXES`) is dropped.
const RUNTIME_TENSOR_PREFIXES: [&str; 4] = ["encoder.", "decoder.", "ctc.", "context_module."];
/// Training-only aux tensors under `context_module.*`: a CTC head over the
/// hotword context embeddings, used only to regularize training. Inference
/// (the deep-biasing fusion in `models::dolphin::hotword_context`) never reads
/// these, so they are dropped to keep the pack lean.
const DROPPED_TENSOR_PREFIXES: [&str; 2] = [
    "context_module.context_decoder.",
    "context_module.context_decoder_ctc_linear.",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(non_camel_case_types)]
pub enum DolphinQuantizationMode {
    /// fp16 weights (rank>=2), f32 for 1-D vectors + CMVN + mel filterbank.
    #[default]
    Fp16,
    /// q8_0 for the rank-2 `.weight` matrices (ne0 % 32 == 0); everything else
    /// keeps its fp16-mode representation.
    Q8_0,
    /// q4_k for the rank-2 `.weight` matrices whose ne0 % 256 == 0, else q8_0;
    /// everything else keeps its fp16-mode representation.
    Q4_K,
}

impl DolphinQuantizationMode {
    /// Canonical lowercase pack-quant tag (`fp16`/`q8_0`/`q4_k`), used to name the
    /// output pack and report the produced rung.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q8_0 => "q8_0",
            Self::Q4_K => "q4_k",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DolphinImportRequest {
    /// The exported full state dict (`full.safetensors`, all-f32).
    pub safetensors_path: PathBuf,
    /// The char vocab (`units.txt`, `token<space>id` per line, id order).
    pub units_path: PathBuf,
    /// Output `.oasr` runtime pack path.
    pub output_path: PathBuf,
    pub model_id: String,
    pub quantization: DolphinQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DolphinImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub vocab_size: usize,
    pub blank_token_id: u32,
}

pub fn convert_local_dolphin_wenet_source_to_runtime_pack(
    request: &DolphinImportRequest,
) -> Result<DolphinImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_path)?;
    let vocab_tokens = read_units_txt(&request.units_path)?;
    let vocab_size = vocab_tokens.len();
    let safetensors = SafetensorsFile::open(&request.safetensors_path)?;

    validate_checkpoint_shape(&safetensors, vocab_size)?;
    assert_every_advertised_dialect_code_resolves(&vocab_tokens)?;

    let mut tensors = build_runtime_tensors(&safetensors, request.quantization)?;
    tensors.push(build_mel_filterbank_tensor());

    let metadata = dolphin_runtime_gguf_metadata(request, &vocab_tokens);
    write_gguf_file_v0(&request.output_path, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "dolphin GGUF writer failed for '{}': {error}",
            request.output_path.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_path).map_err(|error| {
        validate_error(format!(
            "dolphin import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(DolphinImportResult {
        output_path: request.output_path.clone(),
        tensor_count: index.tensors().len(),
        vocab_size,
        blank_token_id: CTC_BLANK_TOKEN_ID,
    })
}

/// Parse `units.txt` (`token<space>id`, one per line) into an id-ordered token
/// list. WeNet char tokens never contain a space, so the id is the trailing
/// whitespace-delimited field.
fn read_units_txt(path: &std::path::Path) -> Result<Vec<String>, LocalSourceImportError> {
    let text = std::fs::read_to_string(path).map_err(|source| LocalSourceImportError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut by_id: BTreeMap<usize, String> = BTreeMap::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let (token, id_str) = line.rsplit_once(char::is_whitespace).ok_or_else(|| {
            validate_error(format!(
                "dolphin units.txt line {} is not '<token> <id>': {line:?}",
                line_no + 1
            ))
        })?;
        let id: usize = id_str.trim().parse().map_err(|error| {
            validate_error(format!(
                "dolphin units.txt line {} has an unparseable id {id_str:?}: {error}",
                line_no + 1
            ))
        })?;
        if by_id.insert(id, token.to_string()).is_some() {
            return Err(validate_error(format!(
                "dolphin units.txt has a duplicate id {id}"
            )));
        }
    }
    let count = by_id.len();
    let mut tokens = Vec::with_capacity(count);
    for expected_id in 0..count {
        let token = by_id.remove(&expected_id).ok_or_else(|| {
            validate_error(format!(
                "dolphin units.txt is missing token id {expected_id}"
            ))
        })?;
        tokens.push(token);
    }
    if tokens.is_empty() {
        return Err(validate_error("dolphin units.txt produced an empty vocab"));
    }
    Ok(tokens)
}

/// Fail closed unless every advertised dialect recognition code (Phase 1's
/// `REGISTERED_DIALECT_CODES` plus the bare `zh` default) builds a full decode
/// prefix against the shipped vocab -- i.e. its `<REGION>` token and the shared
/// `<sos>/<zh>/<asr>/<notimestamp>` control tokens are all present. This is the
/// producer-side guard for the CRITICAL invariant that the picker never advertises
/// a region the executor cannot honor: if a future checkpoint's `units.txt` drops
/// a region token, the import fails here rather than shipping a pack whose picker
/// lies. It runs on the full baked vocab (no single region is baked as *the*
/// default; the region is selected per request at decode time).
fn assert_every_advertised_dialect_code_resolves(
    vocab_tokens: &[String],
) -> Result<(), LocalSourceImportError> {
    for &code in REGISTERED_DIALECT_CODES
        .iter()
        .chain(std::iter::once(&DOLPHIN_DEFAULT_LANGUAGE_CODE))
    {
        build_dolphin_decode_prefix(vocab_tokens, Some(code)).map_err(|error| {
            validate_error(format!(
                "dolphin pack vocab cannot honor advertised dialect code {code:?}: {error}"
            ))
        })?;
    }
    Ok(())
}

/// Fail closed if the checkpoint does not match the small.cn shape the pack
/// metadata will declare (vocab, d_model, layer counts).
fn validate_checkpoint_shape(
    safetensors: &SafetensorsFile,
    vocab_size: usize,
) -> Result<(), LocalSourceImportError> {
    let shape = |name: &str| -> Result<Vec<u64>, LocalSourceImportError> {
        safetensors
            .tensor(name)
            .map(|tensor| tensor.shape.clone())
            .ok_or_else(|| validate_error(format!("dolphin checkpoint missing tensor '{name}'")))
    };
    let expect = |name: &str, actual: &[u64], want: &[u64]| -> Result<(), LocalSourceImportError> {
        if actual == want {
            Ok(())
        } else {
            Err(validate_error(format!(
                "dolphin checkpoint tensor '{name}' shape {actual:?} != expected {want:?}"
            )))
        }
    };

    expect(
        "ctc.ctc_lo.weight",
        &shape("ctc.ctc_lo.weight")?,
        &[vocab_size as u64, ENCODER_D_MODEL as u64],
    )?;
    expect(
        "decoder.output_layer.weight",
        &shape("decoder.output_layer.weight")?,
        &[vocab_size as u64, ENCODER_D_MODEL as u64],
    )?;
    expect(
        "encoder.after_norm.weight",
        &shape("encoder.after_norm.weight")?,
        &[ENCODER_D_MODEL as u64],
    )?;
    expect(
        "encoder.global_cmvn.mean",
        &shape("encoder.global_cmvn.mean")?,
        &[FEATURE_DIM as u64],
    )?;

    let layer_count = |prefix: &str, joint: &str| -> usize {
        let mut seen = BTreeSet::new();
        for tensor in &safetensors.header().tensors {
            if let Some(rest) = tensor.name.strip_prefix(prefix)
                && let Some((idx, _)) = rest.split_once(joint)
                && let Ok(index) = idx.parse::<usize>()
            {
                seen.insert(index);
            }
        }
        seen.len()
    };
    let encoder_layers = layer_count("encoder.encoders.", ".");
    if encoder_layers != ENCODER_N_LAYERS {
        return Err(validate_error(format!(
            "dolphin checkpoint has {encoder_layers} encoder layers, expected {ENCODER_N_LAYERS}"
        )));
    }
    let decoder_layers = layer_count("decoder.decoders.", ".");
    if decoder_layers != DECODER_N_LAYERS {
        return Err(validate_error(format!(
            "dolphin checkpoint has {decoder_layers} decoder layers, expected {DECODER_N_LAYERS}"
        )));
    }

    // Hotword deep-biasing module: only checked when present (older exports
    // without a trained context module still import; the executor's
    // `supports_phrase_bias` degrades to reporting no hotword capability for
    // those packs via the tensor-index probe, not a hard import failure).
    if safetensors
        .tensor("context_module.context_extractor.word_embedding.weight")
        .is_some()
    {
        expect(
            "context_module.context_extractor.word_embedding.weight",
            &shape("context_module.context_extractor.word_embedding.weight")?,
            &[vocab_size as u64, ENCODER_D_MODEL as u64],
        )?;
        expect(
            "context_module.context_extractor.sen_rnn.weight_ih_l0",
            &shape("context_module.context_extractor.sen_rnn.weight_ih_l0")?,
            &[4 * ENCODER_D_MODEL as u64, ENCODER_D_MODEL as u64],
        )?;
        expect(
            "context_module.context_extractor.sen_rnn.weight_ih_l1",
            &shape("context_module.context_extractor.sen_rnn.weight_ih_l1")?,
            &[4 * ENCODER_D_MODEL as u64, 2 * ENCODER_D_MODEL as u64],
        )?;
        expect(
            "context_module.biasing_layer.linear_q.weight",
            &shape("context_module.biasing_layer.linear_q.weight")?,
            &[ENCODER_D_MODEL as u64, ENCODER_D_MODEL as u64],
        )?;
        expect(
            "context_module.combiner.weight",
            &shape("context_module.combiner.weight")?,
            &[ENCODER_D_MODEL as u64, ENCODER_D_MODEL as u64],
        )?;
    }
    Ok(())
}

/// Emit every `encoder.*` / `decoder.*` / `ctc.*` tensor under its exact WeNet
/// name. At fp16, raw element order is preserved (dims == the safetensors shape):
/// rank>=2 weights become f16, everything else (1-D biases/norms, CMVN) stays f32.
/// At q8_0/q4_k, the rank-2 `.weight` matrices are block-quantized (dims reversed
/// so the contiguous `in` axis lands on ne0); all other tensors keep their fp16
/// representation.
fn build_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: DolphinQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for tensor in &safetensors.header().tensors {
        let name = tensor.name.as_str();
        if DROPPED_TENSOR_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            continue;
        }
        if !RUNTIME_TENSOR_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            continue;
        }
        if !seen.insert(name.to_string()) {
            return Err(validate_error(format!(
                "dolphin import mapped duplicate destination tensor '{name}'"
            )));
        }
        let values = decode_safetensors_payload_as_f32(
            &tensor.name,
            &tensor.dtype,
            safetensors.tensor_data(tensor)?,
        )?;
        out.push(make_runtime_tensor(
            name.to_string(),
            tensor.shape.clone(),
            values,
            quantization,
        )?);
    }
    if out.is_empty() {
        return Err(validate_error(
            "dolphin import found no encoder/decoder/ctc tensors in the checkpoint",
        ));
    }
    Ok(out)
}

/// Choose the block-quant type for a rank-2 `.weight` matrix, or `None` to keep
/// its fp16-mode representation. ggml quantizes along ne0, which after the dim
/// reversal (`[out, in]` -> `[in, out]`) is `in` (the safetensors innermost /
/// row-major-contiguous dim). q4_k needs a 256-superblock (`in % 256`), q8_0 a
/// 32-block (`in % 32`); an unaligned `in` falls back to fp16. Only 2-D `.weight`
/// tensors qualify -- convs (rank>2), position tables, and 1-D/CMVN vectors are
/// never quantized.
fn dolphin_quant_type_for_tensor(
    name: &str,
    shape: &[u64],
    quantization: DolphinQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if quantization == DolphinQuantizationMode::Fp16 {
        return None;
    }
    if !name.ends_with(".weight") || shape.len() != 2 {
        return None;
    }
    // Reversed ne0 == the safetensors innermost (last) dim == `in`.
    let ne0 = *shape.last()?;
    if !ne0.is_multiple_of(32) {
        return None;
    }
    if quantization == DolphinQuantizationMode::Q4_K && ne0.is_multiple_of(256) {
        Some(GgufWriteTensorType::Q4_K)
    } else {
        Some(GgufWriteTensorType::Q8_0)
    }
}

/// Build one runtime tensor. Quantizable rank-2 `.weight` matrices are block-
/// quantized with **reversed dims** (ne0 = the contiguous `in` axis); the runtime
/// re-declares its own graph shapes and consumes only the dequantized f32 by
/// element count, so the reversal is transparent to it. Everything else keeps the
/// fp16-mode layout: f16 for rank>=2, f32 for 1-D vectors (biases, norms, the CMVN
/// mean/istd, the rel-pos bias), name-preserving raw element order in both cases.
fn make_runtime_tensor(
    name: String,
    dims: Vec<u64>,
    values: Vec<f32>,
    quantization: DolphinQuantizationMode,
) -> Result<GgufWriteTensor, LocalSourceImportError> {
    if let Some(qtype) = dolphin_quant_type_for_tensor(&name, &dims, quantization) {
        let mut reversed = dims.clone();
        reversed.reverse();
        let data =
            quantize_f32_to_ggml_tensor_data(qtype, &reversed, &values).map_err(|error| {
                validate_error(format!(
                    "dolphin quantization failed for '{name}' ({qtype:?}): {error}"
                ))
            })?;
        return Ok(GgufWriteTensor {
            name,
            dims: reversed,
            tensor_type: qtype,
            data,
        });
    }
    if dims.len() >= 2 {
        let bits: Vec<u16> = values.iter().copied().map(f32_to_f16_bits).collect();
        Ok(GgufWriteTensor {
            name,
            dims,
            tensor_type: GgufWriteTensorType::F16,
            data: encode_f16_bits_le(bits),
        })
    } else {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(GgufWriteTensor {
            name,
            dims,
            tensor_type: GgufWriteTensorType::F32,
            data: bytes,
        })
    }
}

/// Kaldi/HTK mel filterbank `[n_mels, fft_bins]` (peak-normalized triangles, no
/// Slaney area norm) reconstructed from the `train.yaml` fbank config. Stored for
/// the later frontend phase; NOT exercised by the encoder-from-pack load path
/// (which is fed the CMVN'd golden features directly), so it is not yet parity
/// verified against a golden fbank.
fn build_mel_filterbank_tensor() -> GgufWriteTensor {
    let fft_bins = FFT_SIZE / 2 + 1;
    let high_hz = (SAMPLE_RATE_HZ as f32) / 2.0;
    let mel_low = hz_to_mel(MEL_LOW_HZ);
    let mel_high = hz_to_mel(high_hz);
    let mel_delta = (mel_high - mel_low) / (FEATURE_DIM as f32 + 1.0);

    let mut filters = vec![0.0_f32; FEATURE_DIM * fft_bins];
    for mel_idx in 0..FEATURE_DIM {
        let left = mel_low + (mel_idx as f32) * mel_delta;
        let center = mel_low + (mel_idx as f32 + 1.0) * mel_delta;
        let right = mel_low + (mel_idx as f32 + 2.0) * mel_delta;
        for (bin_idx, cell) in filters
            .iter_mut()
            .skip(mel_idx * fft_bins)
            .take(fft_bins)
            .enumerate()
        {
            let hz = (bin_idx as f32) * (SAMPLE_RATE_HZ as f32) / (FFT_SIZE as f32);
            let mel = hz_to_mel(hz);
            let rising = (mel - left) / (center - left);
            let falling = (right - mel) / (right - center);
            *cell = rising.min(falling).max(0.0);
        }
    }
    let mut bytes = Vec::with_capacity(filters.len() * 4);
    for value in &filters {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    GgufWriteTensor {
        name: "dolphin.mel_filters".to_string(),
        dims: vec![FEATURE_DIM as u64, fft_bins as u64],
        tensor_type: GgufWriteTensorType::F32,
        data: bytes,
    }
}

/// Kaldi/HTK mel scale: `mel(f) = 1127 * ln(1 + f / 700)`.
fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn dolphin_runtime_gguf_metadata(
    request: &DolphinImportRequest,
    vocab_tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let vocab_size = vocab_tokens.len();
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str("general.architecture", DOLPHIN_GGML_ARCHITECTURE_ID);
    put_str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put_str(OASR_METADATA_KEY_MODEL_FAMILY, DOLPHIN_MODEL_FAMILY);
    put_str(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        DOLPHIN_GGML_ARCHITECTURE_ID,
    );
    put_str(OASR_METADATA_KEY_AUDIO_FRONTEND, DOLPHIN_AUDIO_FRONTEND_ID);
    put_str(OASR_METADATA_KEY_DECODE_POLICY, DOLPHIN_DECODE_POLICY_ID);
    put_str(GGML_TOKENIZER_ID_KEY, DOLPHIN_TOKENIZER_ID);
    put_str("openasr.model.id", &request.model_id);
    put_str("dolphin.tokenizer.model", "char");

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32("dolphin.encoder.n_layers", ENCODER_N_LAYERS as u32);
    put_u32("dolphin.encoder.d_model", ENCODER_D_MODEL as u32);
    put_u32("dolphin.encoder.n_heads", ENCODER_N_HEADS as u32);
    put_u32("dolphin.encoder.head_dim", ENCODER_HEAD_DIM as u32);
    put_u32("dolphin.encoder.ffn_dim", ENCODER_FFN_DIM as u32);
    put_u32("dolphin.encoder.cgmlp_units", ENCODER_CGMLP_UNITS as u32);
    put_u32("dolphin.encoder.cgmlp_kernel", ENCODER_CGMLP_KERNEL as u32);
    put_u32("dolphin.encoder.merge_kernel", ENCODER_MERGE_KERNEL as u32);
    put_u32("dolphin.encoder.feature_dim", FEATURE_DIM as u32);
    put_u32("dolphin.decoder.n_layers", DECODER_N_LAYERS as u32);
    put_u32("dolphin.decoder.n_heads", DECODER_N_HEADS as u32);
    put_u32("dolphin.decoder.ffn_dim", DECODER_FFN_DIM as u32);
    put_u32("dolphin.decoder.max_ctx", DECODER_MAX_CTX as u32);
    put_u32("dolphin.vocab_size", vocab_size as u32);
    put_u32("dolphin.sos_token_id", SOS_TOKEN_ID);
    put_u32("dolphin.eos_token_id", EOS_TOKEN_ID);
    put_u32("ctc.blank_token_id", CTC_BLANK_TOKEN_ID);
    // fbank frontend config (the mel filterbank contract for the later phase).
    put_u32("dolphin.audio.sample_rate", SAMPLE_RATE_HZ);
    put_u32("dolphin.audio.n_fft", FFT_SIZE as u32);
    put_u32("dolphin.audio.frame_length_ms", FRAME_LENGTH_MS);
    put_u32("dolphin.audio.frame_shift_ms", FRAME_SHIFT_MS);
    put_u32("dolphin.audio.n_mels", FEATURE_DIM as u32);

    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufWriteValue::StringArray(vocab_tokens.to_vec()),
    );
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use DolphinQuantizationMode::Fp16;

    fn string_metadata(metadata: &BTreeMap<String, GgufWriteValue>, key: &str) -> Option<String> {
        match metadata.get(key) {
            Some(GgufWriteValue::String(value)) => Some(value.clone()),
            _ => None,
        }
    }

    fn u32_metadata(metadata: &BTreeMap<String, GgufWriteValue>, key: &str) -> Option<u32> {
        match metadata.get(key) {
            Some(GgufWriteValue::U32(value)) => Some(*value),
            _ => None,
        }
    }

    fn fixture_request() -> DolphinImportRequest {
        DolphinImportRequest {
            safetensors_path: PathBuf::from("/tmp/dolphin/full.safetensors"),
            units_path: PathBuf::from("/tmp/dolphin/units.txt"),
            output_path: PathBuf::from("/tmp/dolphin-out.oasr"),
            model_id: "dolphin-cn-dialect-small".to_string(),
            quantization: DolphinQuantizationMode::Fp16,
        }
    }

    #[test]
    fn runtime_metadata_declares_dolphin_selection_and_contract_keys() {
        let tokens: Vec<String> = (0..18173).map(|i| format!("t{i}")).collect();
        let metadata = dolphin_runtime_gguf_metadata(&fixture_request(), &tokens);
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_FAMILY),
            Some(DOLPHIN_MODEL_FAMILY.to_string())
        );
        assert_eq!(
            string_metadata(&metadata, OASR_METADATA_KEY_MODEL_ARCHITECTURE),
            Some(DOLPHIN_GGML_ARCHITECTURE_ID.to_string())
        );
        assert_eq!(
            string_metadata(&metadata, GGML_TOKENIZER_ID_KEY),
            Some(DOLPHIN_TOKENIZER_ID.to_string())
        );
        assert_eq!(u32_metadata(&metadata, "dolphin.vocab_size"), Some(18173));
        assert_eq!(u32_metadata(&metadata, "ctc.blank_token_id"), Some(0));
        assert_eq!(
            u32_metadata(&metadata, "dolphin.encoder.d_model"),
            Some(768)
        );
    }

    /// The full special-token block every advertised dialect code needs
    /// (control tokens + one `<REGION>` token per registered code plus `<CN>`).
    fn vocab_with_all_dialect_tokens() -> Vec<String> {
        let mut tokens: Vec<String> = ["<sos>", "<eos>", "<asr>", "<zh>", "<notimestamp>", "<CN>"]
            .iter()
            .map(|token| token.to_string())
            .collect();
        for &code in REGISTERED_DIALECT_CODES {
            let region = crate::models::dolphin::language::dolphin_region_token_for_code(code)
                .expect("registered code maps to a region token");
            tokens.push(region.to_string());
        }
        tokens
    }

    #[test]
    fn advertised_dialect_codes_resolve_against_full_vocab() {
        // A vocab carrying every region + control token clears the producer guard.
        let vocab = vocab_with_all_dialect_tokens();
        assert_every_advertised_dialect_code_resolves(&vocab).expect("all codes resolve");
    }

    #[test]
    fn missing_region_token_fails_the_import_guard() {
        // Drop `<SICHUAN>` (an advertised region) and the import must fail closed
        // rather than ship a pack whose picker advertises a region it can't honor.
        let mut vocab = vocab_with_all_dialect_tokens();
        vocab.retain(|token| token != "<SICHUAN>");
        let error = assert_every_advertised_dialect_code_resolves(&vocab)
            .expect_err("missing region must fail the guard");
        let message = error.to_string();
        assert!(
            message.contains("zh-sichuan"),
            "guard error should name the unhonored code, got: {message}"
        );
    }

    #[test]
    fn quant_type_selection_matches_alignment_and_mode() {
        // fp16 mode never quantizes.
        assert_eq!(
            dolphin_quant_type_for_tensor("decoder.output_layer.weight", &[18173, 768], Fp16),
            None
        );
        // q8_0: any rank-2 `.weight` whose in-dim (last, -> reversed ne0) is 32-aligned.
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "decoder.output_layer.weight",
                &[18173, 768],
                DolphinQuantizationMode::Q8_0
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        // q4_k: in-dim 256-aligned -> Q4_K; only 32- (not 256-) aligned -> Q8_0.
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "encoder.encoders.0.feed_forward.w_2.weight",
                &[768, 3072],
                DolphinQuantizationMode::Q4_K
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "some.proj.weight",
                &[10, 96],
                DolphinQuantizationMode::Q4_K
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        // Unaligned in-dim, non-`.weight`, rank!=2, and 1-D tensors are never quantized.
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "some.proj.weight",
                &[10, 100],
                DolphinQuantizationMode::Q8_0
            ),
            None
        );
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "encoder.embed.pos_enc.pe",
                &[1, 5000, 768],
                DolphinQuantizationMode::Q8_0
            ),
            None
        );
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "encoder.embed.conv.0.bias",
                &[768],
                DolphinQuantizationMode::Q8_0
            ),
            None
        );
        assert_eq!(
            dolphin_quant_type_for_tensor(
                "encoder.embed.conv.2.weight",
                &[768, 768, 3, 3],
                DolphinQuantizationMode::Q8_0
            ),
            None
        );
    }

    #[test]
    fn fp16_tensor_layout_is_unchanged_by_the_quant_plumbing() {
        // A rank-2 weight in fp16 mode stays f16 with raw (non-reversed) dims.
        let weight = make_runtime_tensor("x.weight".to_string(), vec![4, 2], vec![1.0; 8], Fp16)
            .expect("fp16 weight");
        assert_eq!(weight.tensor_type, GgufWriteTensorType::F16);
        assert_eq!(weight.dims, vec![4, 2]);
        assert_eq!(weight.data.len(), 8 * 2); // f16 = 2 bytes/elem, no reversal.
        // A 1-D vector stays f32.
        let bias = make_runtime_tensor("x.bias".to_string(), vec![4], vec![1.0; 4], Fp16)
            .expect("fp16 bias");
        assert_eq!(bias.tensor_type, GgufWriteTensorType::F32);
        assert_eq!(bias.dims, vec![4]);
    }

    #[test]
    fn quantized_weight_reverses_dims_to_block_align_ne0() {
        // in-dim 768 (256-aligned) -> q4_k, dims reversed so ne0 == 768.
        let weight = make_runtime_tensor(
            "decoder.output_layer.weight".to_string(),
            vec![18173, 768],
            vec![0.1; 18173 * 768],
            DolphinQuantizationMode::Q4_K,
        )
        .expect("quantized weight");
        assert_eq!(weight.tensor_type, GgufWriteTensorType::Q4_K);
        assert_eq!(weight.dims, vec![768, 18173]);
    }

    #[test]
    fn mel_filterbank_has_expected_shape_and_is_bounded() {
        let tensor = build_mel_filterbank_tensor();
        let fft_bins = FFT_SIZE / 2 + 1;
        assert_eq!(tensor.dims, vec![FEATURE_DIM as u64, fft_bins as u64]);
        assert_eq!(tensor.data.len(), FEATURE_DIM * fft_bins * 4);
        // Peak-normalized triangles sit within [0, 1].
        for chunk in tensor.data.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert!(
                (0.0..=1.0001).contains(&value),
                "mel weight out of range: {value}"
            );
        }
    }
}
