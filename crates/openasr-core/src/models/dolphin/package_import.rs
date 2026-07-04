//! Convert a local Dolphin WeNet checkpoint (exported `full.safetensors` +
//! `units.txt` char vocab, with `global_cmvn` folded into the encoder tensors)
//! into an OpenASR `.oasr` (GGUF-v0) runtime pack at fp16.
//!
//! Naming contract: the encoder/decoder/CTC tensors are stored under their
//! **exact WeNet state-dict names** in raw element order (only f32 -> f16 for the
//! rank>=2 weight matrices/convs; the 1-D biases/norms and the CMVN vectors stay
//! f32). This keeps the runtime executor trivial and, crucially, lets it feed the
//! already parity-verified `encoder_graph::encode()` byte-for-byte the same tensor
//! buffers the raw-safetensors parity harness used — the only delta is the fp16
//! rounding of the weights.
//!
//! Baked into the pack:
//!   * every `encoder.*` / `decoder.*` / `ctc.*` tensor (the `context_module.*`
//!     hotword-biasing tensors are intentionally dropped — not part of the core
//!     ASR forward),
//!   * the global CMVN mean/istd (already present as `encoder.global_cmvn.*`),
//!   * a kaldi/HTK mel filterbank (`dolphin.mel_filters`) reconstructed from the
//!     `train.yaml` fbank config for the later frontend phase,
//!   * the char tokenizer (`tokenizer.ggml.tokens`, ids in `units.txt` order),
//!   * the runtime scalar contract keys the install gate validates.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use crate::arch::{
    DOLPHIN_AUDIO_FRONTEND_ID, DOLPHIN_DECODE_POLICY_ID, DOLPHIN_GGML_ARCHITECTURE_ID,
    DOLPHIN_MODEL_FAMILY, DOLPHIN_TOKENIZER_ID,
};
use crate::ggml_runtime::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, read_gguf_tensor_index,
    write_gguf_file_v0,
};
use crate::models::ggml_family_adapter::GGML_TOKENIZER_ID_KEY;
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

/// The reference Sichuan decode prefix `<sos> <zh> <SICHUAN> <asr> <notimestamp>`
/// (OWSM-style: sos + lang + region + task + timestamp). Baked for the later
/// decode phase; not consumed by the encoder-from-pack load path.
const SICHUAN_PREFIX_TOKEN_IDS: [u32; 5] = [2, 5, 10, 4, 109];

/// WeNet state-dict namespaces baked into the runtime pack (in order).
const RUNTIME_TENSOR_PREFIXES: [&str; 3] = ["encoder.", "decoder.", "ctc."];
/// Hotword contextual-biasing module: present in the checkpoint but not part of
/// the core ASR forward, so it is dropped from the runtime pack.
const DROPPED_TENSOR_PREFIX: &str = "context_module.";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DolphinQuantizationMode {
    /// fp16 weights (rank>=2), f32 for 1-D vectors + CMVN + mel filterbank.
    #[default]
    Fp16,
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

    let mut tensors = build_runtime_tensors(&safetensors)?;
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
    Ok(())
}

/// Emit every `encoder.*` / `decoder.*` / `ctc.*` tensor under its exact WeNet
/// name, raw element order preserved (dims == the safetensors shape). rank>=2
/// weights become f16, everything else (1-D biases/norms, CMVN) stays f32.
fn build_runtime_tensors(
    safetensors: &SafetensorsFile,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for tensor in &safetensors.header().tensors {
        let name = tensor.name.as_str();
        if name.starts_with(DROPPED_TENSOR_PREFIX) {
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
        ));
    }
    if out.is_empty() {
        return Err(validate_error(
            "dolphin import found no encoder/decoder/ctc tensors in the checkpoint",
        ));
    }
    Ok(out)
}

/// f16 for rank>=2 weight matrices and convs; f32 for 1-D vectors (biases,
/// norms, the CMVN mean/istd, the rel-pos bias). The name-preserving raw element
/// order is kept in both cases.
fn make_runtime_tensor(name: String, dims: Vec<u64>, values: Vec<f32>) -> GgufWriteTensor {
    if dims.len() >= 2 {
        let bits: Vec<u16> = values.iter().copied().map(f32_to_f16_bits).collect();
        GgufWriteTensor {
            name,
            dims,
            tensor_type: GgufWriteTensorType::F16,
            data: encode_f16_bits_le(bits),
        }
    } else {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        GgufWriteTensor {
            name,
            dims,
            tensor_type: GgufWriteTensorType::F32,
            data: bytes,
        }
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
        "dolphin.prompt.prefix_token_ids".to_string(),
        GgufWriteValue::U32Array(SICHUAN_PREFIX_TOKEN_IDS.to_vec()),
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
