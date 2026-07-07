//! Convert a local `FireRedTeam/FireRedASR-AED-L` source (a `model.safetensors`
//! produced by `pt_to_safetensors.py` from the checkpoint's `model.pth.tar`,
//! plus the char+SPM `dict.txt` vocab and the kaldi-text `cmvn.txt` stats) into
//! an OpenASR `.oasr` (GGUF-v0) runtime pack.
//!
//! Mirrors `sensevoice::package_import` (the same safetensors -> GGUF path).
//! Layer mapping: `encoder.layer_stack.{i}` becomes `enc.blk.{i}` (16 Conformer
//! blocks) and `decoder.layer_stack.{i}` becomes `dec.blk.{i}` (16 pre-norm
//! Transformer decoder blocks). The Conv2d subsampling stem maps to
//! `enc.subsample.*`, the sinusoidal position tables to `enc.pos_enc.pe` /
//! `dec.pos_enc.pe`, and the token embedding / output projection to
//! `dec.tok_emb.weight` / `dec.out_proj.weight`. The upstream module ties
//! `tgt_word_prj.weight = tgt_word_emb.weight`, but the checkpoint stores both
//! tensors; each is loaded from its own on-disk bytes (never re-tied here).
//!
//! Upstream architecture facts this importer encodes (verified against the
//! FireRedASR source, `fireredasr/models/module/{conformer_encoder,
//! transformer_decoder}.py`):
//!   * encoder MHSA has NO projection biases (`w_qs/w_ks/w_vs/fc/linear_pos`
//!     are all `bias=False`) and normalizes q/k/v with THREE separate
//!     post-input LayerNorms (`layer_norm_q/k/v`), not one shared pre-norm;
//!   * the conv module's `batch_norm` is actually an `nn.LayerNorm(2*d_model)`
//!     (no running stats in the checkpoint), so there is NO BatchNorm fold;
//!   * decoder self/cross attention: `w_qs`/`w_vs`/`fc` carry biases, `w_ks`
//!     does not; the FFN activation is GELU;
//!   * CMVN is plain kaldi accumulator stats (`cmvn.txt`, 2 x (dim+1)):
//!     `mean = sum/count`, `var = sumsq/count - mean^2` floored at 1e-20,
//!     baked as `frontend.cmvn.neg_mean` / `frontend.cmvn.inv_stddev`;
//!   * the fbank frontend is kaldi-style 80-mel / 25 ms / 10 ms / 16 kHz with
//!     dither 0 and snip_edges (upstream `kaldi_native_fbank` defaults).
//!
//! Keep-quantized: 2-D linear projections (attention q/k/v/out/pos, macaron
//! FFN up/down, conv pointwise (kernel-1 squeezed), decoder projections,
//! `dec.out_proj.weight`) quantize to q8_0/q4_k. Norms, biases, the flattened
//! rel-pos bias vectors, the CMVN vectors, and the mel filterbank stay f32;
//! the depthwise/subsampling conv kernels, position tables, and the token
//! embedding (get_rows source) stay f16.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::arch::{
    FIRERED_AED_AUDIO_FRONTEND_ID, FIRERED_AED_DECODE_POLICY_ID, FIRERED_AED_GGML_ARCHITECTURE_ID,
    FIRERED_AED_MODEL_FAMILY, FIRERED_AED_TOKENIZER_ID,
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

const SOURCE_MODEL_SAFETENSORS: &str = "model.safetensors";
const SOURCE_DICT_TXT: &str = "dict.txt";
const SOURCE_CMVN_TXT: &str = "cmvn.txt";

// fbank frontend contract (upstream `kaldi_native_fbank` defaults with
// `num_mel_bins=80, frame_length=25, frame_shift=10, dither=0`): 16 kHz mono,
// 400-sample window rounded up to a 512-point FFT, kaldi/HTK mel scale from
// 20 Hz to Nyquist.
const SAMPLE_RATE_HZ: u32 = 16_000;
const FRAME_LENGTH_MS: u32 = 25;
const FRAME_SHIFT_MS: u32 = 10;
const FFT_SIZE: usize = 512;
const MEL_LOW_HZ: f32 = 20.0;

/// Kaldi CMVN variance floor (upstream `fireredasr/data/asr_feat.py`).
const CMVN_VARIANCE_FLOOR: f64 = 1e-20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum FireRedAedQuantizationMode {
    #[default]
    Fp16,
    Q8_0,
    Q4_K,
}

impl FireRedAedQuantizationMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Q8_0 => "q8_0",
            Self::Q4_K => "q4_k",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FireRedAedImportRequest {
    /// Source directory containing `model.safetensors`, `dict.txt`, `cmvn.txt`.
    pub source_root: PathBuf,
    /// Output path for one runtime pack file (`.oasr`).
    pub output_root: PathBuf,
    pub model_id: String,
    pub quantization: FireRedAedQuantizationMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FireRedAedImportResult {
    pub output_path: PathBuf,
    pub tensor_count: usize,
    pub vocab_size: usize,
}

/// Architecture facts derived from the safetensors shapes (fail-closed:
/// inconsistent shapes reject the import rather than writing a broken pack).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FireRedAedDerivedHparams {
    encoder_n_layers: usize,
    decoder_n_layers: usize,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    encoder_ffn_dim: usize,
    decoder_ffn_dim: usize,
    conv_kernel: usize,
    subsample_channels: usize,
    subsample_out_dim: usize,
    feature_dim: usize,
    encoder_pe_len: usize,
    decoder_pe_len: usize,
    vocab_size: usize,
    sos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,
}

pub fn convert_local_firered_aed_source_to_runtime_pack(
    request: &FireRedAedImportRequest,
) -> Result<FireRedAedImportResult, LocalSourceImportError> {
    validate_output_pack_extension(&request.output_root)?;
    let safetensors = SafetensorsFile::open(request.source_root.join(SOURCE_MODEL_SAFETENSORS))?;
    let vocab_tokens = read_dict_txt(&request.source_root.join(SOURCE_DICT_TXT))?;
    let (cmvn_neg_mean, cmvn_inv_stddev) =
        parse_kaldi_cmvn_stats(&request.source_root.join(SOURCE_CMVN_TXT))?;

    let hparams = derive_and_validate_hparams(&safetensors, &vocab_tokens, cmvn_neg_mean.len())?;

    let mut tensors = build_firered_runtime_tensors(&safetensors, request.quantization)?;
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
    tensors.push(build_mel_filterbank_tensor(hparams.feature_dim));

    let metadata = firered_runtime_gguf_metadata(&hparams, request, &vocab_tokens);
    write_gguf_file_v0(&request.output_root, &metadata, &tensors).map_err(|error| {
        validate_error(format!(
            "firered-aed GGUF writer failed for '{}': {error}",
            request.output_root.display()
        ))
    })?;

    let index = read_gguf_tensor_index(&request.output_root).map_err(|error| {
        validate_error(format!(
            "firered-aed import produced an unreadable tensor index: {error}"
        ))
    })?;
    Ok(FireRedAedImportResult {
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

/// Parse `dict.txt` (`token<space>id`, one per line, ids 0..N contiguous) into
/// an id-ordered token list. Mirrors upstream `TokenDict`: a `<space>` token
/// becomes a literal `" "`. FireRed tokens themselves never contain whitespace,
/// so the id is the trailing whitespace-delimited field.
fn read_dict_txt(path: &Path) -> Result<Vec<String>, LocalSourceImportError> {
    let text = std::fs::read_to_string(path).map_err(|source| LocalSourceImportError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let mut by_id: BTreeMap<usize, String> = BTreeMap::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        let (token, id_str) = line.rsplit_once(char::is_whitespace).ok_or_else(|| {
            validate_error(format!(
                "firered-aed dict.txt line {} is not '<token> <id>': {line:?}",
                line_no + 1
            ))
        })?;
        let id: usize = id_str.trim().parse().map_err(|error| {
            validate_error(format!(
                "firered-aed dict.txt line {} has an unparseable id {id_str:?}: {error}",
                line_no + 1
            ))
        })?;
        let token = if token == "<space>" { " " } else { token };
        if by_id.insert(id, token.to_string()).is_some() {
            return Err(validate_error(format!(
                "firered-aed dict.txt has a duplicate id {id}"
            )));
        }
    }
    let count = by_id.len();
    let mut tokens = Vec::with_capacity(count);
    for expected_id in 0..count {
        let token = by_id.remove(&expected_id).ok_or_else(|| {
            validate_error(format!(
                "firered-aed dict.txt is missing token id {expected_id}"
            ))
        })?;
        tokens.push(token);
    }
    if tokens.is_empty() {
        return Err(validate_error(
            "firered-aed dict.txt produced an empty vocab",
        ));
    }
    Ok(tokens)
}

/// Parse a kaldi text CMVN stats matrix (`cmvn.txt`, `[ row0 \n row1 ]` with
/// 2 x (dim+1) values: sums + frame count in row 0, sum-of-squares + 0 in
/// row 1) into `(neg_mean, inv_stddev)` f32 vectors, using the exact upstream
/// formula (`fireredasr/data/asr_feat.py::CMVN`).
fn parse_kaldi_cmvn_stats(path: &Path) -> Result<(Vec<f32>, Vec<f32>), LocalSourceImportError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        validate_error(format!(
            "firered-aed import cannot read '{}': {error}",
            path.display()
        ))
    })?;
    let open = text
        .find('[')
        .ok_or_else(|| validate_error("firered-aed cmvn.txt has no '[' matrix opener"))?;
    let close = text
        .rfind(']')
        .ok_or_else(|| validate_error("firered-aed cmvn.txt has no ']' matrix closer"))?;
    if close < open {
        return Err(validate_error("firered-aed cmvn.txt has ']' before '['"));
    }
    let body = &text[open + 1..close];
    let rows: Vec<Vec<f64>> = body
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|token| {
                    token.parse::<f64>().map_err(|error| {
                        validate_error(format!(
                            "firered-aed cmvn.txt has a non-numeric value '{token}': {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<f64>, _>>()
        })
        .collect::<Result<Vec<Vec<f64>>, _>>()?;
    if rows.len() != 2 {
        return Err(validate_error(format!(
            "firered-aed cmvn.txt must contain exactly 2 stat rows, found {}",
            rows.len()
        )));
    }
    let (sums, sum_squares) = (&rows[0], &rows[1]);
    if sums.len() != sum_squares.len() || sums.len() < 2 {
        return Err(validate_error(format!(
            "firered-aed cmvn.txt row lengths are inconsistent ({} vs {})",
            sums.len(),
            sum_squares.len()
        )));
    }
    let dim = sums.len() - 1;
    let count = sums[dim];
    if count < 1.0 {
        return Err(validate_error(format!(
            "firered-aed cmvn.txt frame count {count} must be >= 1"
        )));
    }
    let mut neg_mean = Vec::with_capacity(dim);
    let mut inv_stddev = Vec::with_capacity(dim);
    for d in 0..dim {
        let mean = sums[d] / count;
        let variance = (sum_squares[d] / count - mean * mean).max(CMVN_VARIANCE_FLOOR);
        neg_mean.push((-mean) as f32);
        inv_stddev.push((1.0 / variance.sqrt()) as f32);
    }
    Ok((neg_mean, inv_stddev))
}

fn derive_and_validate_hparams(
    safetensors: &SafetensorsFile,
    vocab_tokens: &[String],
    cmvn_dim: usize,
) -> Result<FireRedAedDerivedHparams, LocalSourceImportError> {
    let mut shape_by_name: BTreeMap<&str, &[u64]> = BTreeMap::new();
    for tensor in &safetensors.header().tensors {
        shape_by_name.insert(tensor.name.as_str(), tensor.shape.as_slice());
    }
    let shape = |name: &str| -> Result<&[u64], LocalSourceImportError> {
        shape_by_name
            .get(name)
            .copied()
            .ok_or_else(|| validate_error(format!("firered-aed source is missing tensor '{name}'")))
    };
    let expect = |name: &str, want: &[u64]| -> Result<(), LocalSourceImportError> {
        let actual = shape(name)?;
        if actual == want {
            Ok(())
        } else {
            Err(validate_error(format!(
                "firered-aed tensor '{name}' shape {actual:?} != expected {want:?}"
            )))
        }
    };

    // Contiguous layer counts (fail-closed on gaps: the count stops at the
    // first missing index, and any tensor beyond it later fails the
    // unmapped-tensor guard in `build_firered_runtime_tensors`).
    let count_layers = |prefix: &str, probe: &str| -> usize {
        let mut count = 0usize;
        while shape_by_name.contains_key(format!("{prefix}.{count}.{probe}").as_str()) {
            count += 1;
        }
        count
    };
    let encoder_n_layers = count_layers("encoder.layer_stack", "mhsa.w_qs.weight");
    if encoder_n_layers == 0 {
        return Err(validate_error(
            "firered-aed source has no 'encoder.layer_stack.N.*' tensors",
        ));
    }
    let decoder_n_layers = count_layers("decoder.layer_stack", "self_attn.w_qs.weight");
    if decoder_n_layers == 0 {
        return Err(validate_error(
            "firered-aed source has no 'decoder.layer_stack.N.*' tensors",
        ));
    }

    let prj = shape("decoder.tgt_word_prj.weight")?;
    let emb = shape("decoder.tgt_word_emb.weight")?;
    if prj.len() != 2 || emb.len() != 2 {
        return Err(validate_error(
            "firered-aed embedding/projection tensors must be rank 2",
        ));
    }
    let vocab_size = prj[0] as usize;
    let d_model = prj[1] as usize;
    if emb[0] as usize != vocab_size || emb[1] as usize != d_model {
        return Err(validate_error(format!(
            "firered-aed tgt_word_emb shape {emb:?} != tgt_word_prj shape {prj:?}"
        )));
    }
    if vocab_size != vocab_tokens.len() {
        return Err(validate_error(format!(
            "firered-aed output projection vocab {vocab_size} != dict.txt tokens {}",
            vocab_tokens.len()
        )));
    }

    let pos_bias = shape("encoder.layer_stack.0.mhsa.pos_bias_u")?;
    let (n_heads, head_dim) = match pos_bias {
        [heads, head_dim] => (*heads as usize, *head_dim as usize),
        _ => {
            return Err(validate_error(format!(
                "firered-aed 'encoder.layer_stack.0.mhsa.pos_bias_u' has shape {pos_bias:?}, expected rank 2"
            )));
        }
    };
    if n_heads == 0 || n_heads * head_dim != d_model {
        return Err(validate_error(format!(
            "firered-aed heads {n_heads} * head_dim {head_dim} != d_model {d_model}"
        )));
    }

    let enc_ffn = shape("encoder.layer_stack.0.ffn1.net.1.weight")?;
    let dec_ffn = shape("decoder.layer_stack.0.mlp.w_1.weight")?;
    if enc_ffn.len() != 2
        || dec_ffn.len() != 2
        || enc_ffn[1] as usize != d_model
        || dec_ffn[1] as usize != d_model
    {
        return Err(validate_error(
            "firered-aed FFN up-projections must be rank 2 with d_model input",
        ));
    }
    let encoder_ffn_dim = enc_ffn[0] as usize;
    let decoder_ffn_dim = dec_ffn[0] as usize;

    let dw = shape("encoder.layer_stack.0.conv.depthwise_conv.weight")?;
    let conv_kernel = match dw {
        [channels, one, kernel] if *channels as usize == 2 * d_model && *one == 1 => {
            *kernel as usize
        }
        _ => {
            return Err(validate_error(format!(
                "firered-aed depthwise conv has unexpected shape {dw:?} (want [{}, 1, k])",
                2 * d_model
            )));
        }
    };
    if conv_kernel % 2 != 1 {
        return Err(validate_error(format!(
            "firered-aed conv kernel {conv_kernel} must be odd (symmetric padding)"
        )));
    }
    expect(
        "encoder.layer_stack.0.conv.pointwise_conv1.weight",
        &[4 * d_model as u64, d_model as u64, 1],
    )?;
    expect(
        "encoder.layer_stack.0.conv.pointwise_conv2.weight",
        &[d_model as u64, 2 * d_model as u64, 1],
    )?;
    expect(
        "encoder.layer_stack.0.conv.batch_norm.weight",
        &[2 * d_model as u64],
    )?;

    // Conv2d subsampling stem: two k=3 s=2 convs, then a linear from the
    // flattened `channels * ((feat-1)/2-1)/2` features to d_model.
    let conv1 = shape("encoder.input_preprocessor.conv.0.weight")?;
    let subsample_channels = match conv1 {
        [channels, 1, 3, 3] => *channels as usize,
        _ => {
            return Err(validate_error(format!(
                "firered-aed subsampling conv1 has unexpected shape {conv1:?} (want [C, 1, 3, 3])"
            )));
        }
    };
    expect(
        "encoder.input_preprocessor.conv.2.weight",
        &[subsample_channels as u64, subsample_channels as u64, 3, 3],
    )?;
    let feature_dim = cmvn_dim;
    let expected_subsample_out = subsample_channels * (((feature_dim - 1) / 2 - 1) / 2);
    let out = shape("encoder.input_preprocessor.out.weight")?;
    if out.len() != 2 || out[0] as usize != d_model {
        return Err(validate_error(format!(
            "firered-aed subsampling out projection has unexpected shape {out:?}"
        )));
    }
    let subsample_out_dim = out[1] as usize;
    if subsample_out_dim != expected_subsample_out {
        return Err(validate_error(format!(
            "firered-aed subsampling out dim {subsample_out_dim} != {subsample_channels} channels x \
             subsampled {feature_dim}-mel width ({expected_subsample_out})"
        )));
    }

    let enc_pe = shape("encoder.positional_encoding.pe")?;
    let encoder_pe_len = match enc_pe {
        [1, len, dm] if *dm as usize == d_model => *len as usize,
        _ => {
            return Err(validate_error(format!(
                "firered-aed encoder pe has unexpected shape {enc_pe:?} (want [1, len, {d_model}])"
            )));
        }
    };
    // Relative position table: 2 * max_len - 1 rows (odd by construction).
    if encoder_pe_len % 2 != 1 {
        return Err(validate_error(format!(
            "firered-aed encoder rel-pos table length {encoder_pe_len} must be odd (2*max-1)"
        )));
    }
    let dec_pe = shape("decoder.positional_encoding.pe")?;
    let decoder_pe_len = match dec_pe {
        [1, len, dm] if *dm as usize == d_model => *len as usize,
        _ => {
            return Err(validate_error(format!(
                "firered-aed decoder pe has unexpected shape {dec_pe:?} (want [1, len, {d_model}])"
            )));
        }
    };

    let required_token = |content: &str| -> Result<u32, LocalSourceImportError> {
        vocab_tokens
            .iter()
            .position(|token| token == content)
            .map(|index| index as u32)
            .ok_or_else(|| validate_error(format!("firered-aed dict.txt has no '{content}' token")))
    };
    let sos_token_id = required_token("<sos>")?;
    let eos_token_id = required_token("<eos>")?;
    let pad_token_id = required_token("<pad>")?;

    Ok(FireRedAedDerivedHparams {
        encoder_n_layers,
        decoder_n_layers,
        d_model,
        n_heads,
        head_dim,
        encoder_ffn_dim,
        decoder_ffn_dim,
        conv_kernel,
        subsample_channels,
        subsample_out_dim,
        feature_dim,
        encoder_pe_len,
        decoder_pe_len,
        vocab_size,
        sos_token_id,
        eos_token_id,
        pad_token_id,
    })
}

/// Storage class for one mapped tensor, deciding dtype + dim handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorClass {
    /// 1-D norms/biases: f32, dims unchanged.
    F32Vector,
    /// Rel-pos bias `[heads, head_dim]`: flattened to 1-D `[d_model]` f32
    /// (the graph reshapes it, matching the cohere conformer convention).
    PosBiasFlatten,
    /// 2-D `mul_mat` projection: reversed dims, quantizable.
    Linear,
    /// Pointwise conv `[out, in, 1]`: trailing kernel-1 squeezed, then treated
    /// as a Linear (`[in, out]` reversed, quantizable).
    PointwiseConvSqueeze,
    /// Conv kernels (subsampling Conv2d, depthwise Conv1d): reversed dims, f16
    /// (ggml im2col/conv_2d_dw take f16 kernels), never quantized.
    ConvKernel,
    /// Position tables / token embedding: reversed dims, f16, never quantized.
    F16Table,
}

/// Map a FireRed state-dict tensor name to its `.oasr` target name + storage
/// class. Returns `None` for names this importer does not recognize; the
/// caller fails closed on those (the AED checkpoint has no droppable tensors).
fn map_firered_tensor_name(source_name: &str) -> Option<(String, TensorClass)> {
    use TensorClass::*;
    match source_name {
        "encoder.input_preprocessor.conv.0.weight" => {
            return Some(("enc.subsample.conv1.weight".into(), ConvKernel));
        }
        "encoder.input_preprocessor.conv.0.bias" => {
            return Some(("enc.subsample.conv1.bias".into(), F32Vector));
        }
        "encoder.input_preprocessor.conv.2.weight" => {
            return Some(("enc.subsample.conv2.weight".into(), ConvKernel));
        }
        "encoder.input_preprocessor.conv.2.bias" => {
            return Some(("enc.subsample.conv2.bias".into(), F32Vector));
        }
        "encoder.input_preprocessor.out.weight" => {
            return Some(("enc.subsample.out.weight".into(), Linear));
        }
        "encoder.input_preprocessor.out.bias" => {
            return Some(("enc.subsample.out.bias".into(), F32Vector));
        }
        "encoder.positional_encoding.pe" => return Some(("enc.pos_enc.pe".into(), F16Table)),
        "decoder.positional_encoding.pe" => return Some(("dec.pos_enc.pe".into(), F16Table)),
        "decoder.tgt_word_emb.weight" => return Some(("dec.tok_emb.weight".into(), F16Table)),
        "decoder.tgt_word_prj.weight" => return Some(("dec.out_proj.weight".into(), Linear)),
        "decoder.layer_norm_out.weight" => return Some(("dec.out_norm.weight".into(), F32Vector)),
        "decoder.layer_norm_out.bias" => return Some(("dec.out_norm.bias".into(), F32Vector)),
        _ => {}
    }
    if let Some(rest) = source_name.strip_prefix("encoder.layer_stack.") {
        let (layer, tail) = rest.split_once('.')?;
        let layer: usize = layer.parse().ok()?;
        let (suffix, class) = match tail {
            "ffn1.net.0.weight" => ("ffn1.norm.weight", F32Vector),
            "ffn1.net.0.bias" => ("ffn1.norm.bias", F32Vector),
            "ffn1.net.1.weight" => ("ffn1.up.weight", Linear),
            "ffn1.net.1.bias" => ("ffn1.up.bias", F32Vector),
            "ffn1.net.4.weight" => ("ffn1.down.weight", Linear),
            "ffn1.net.4.bias" => ("ffn1.down.bias", F32Vector),
            "ffn2.net.0.weight" => ("ffn2.norm.weight", F32Vector),
            "ffn2.net.0.bias" => ("ffn2.norm.bias", F32Vector),
            "ffn2.net.1.weight" => ("ffn2.up.weight", Linear),
            "ffn2.net.1.bias" => ("ffn2.up.bias", F32Vector),
            "ffn2.net.4.weight" => ("ffn2.down.weight", Linear),
            "ffn2.net.4.bias" => ("ffn2.down.bias", F32Vector),
            "mhsa.layer_norm_q.weight" => ("attn.norm_q.weight", F32Vector),
            "mhsa.layer_norm_q.bias" => ("attn.norm_q.bias", F32Vector),
            "mhsa.layer_norm_k.weight" => ("attn.norm_k.weight", F32Vector),
            "mhsa.layer_norm_k.bias" => ("attn.norm_k.bias", F32Vector),
            "mhsa.layer_norm_v.weight" => ("attn.norm_v.weight", F32Vector),
            "mhsa.layer_norm_v.bias" => ("attn.norm_v.bias", F32Vector),
            "mhsa.w_qs.weight" => ("attn.q.weight", Linear),
            "mhsa.w_ks.weight" => ("attn.k.weight", Linear),
            "mhsa.w_vs.weight" => ("attn.v.weight", Linear),
            "mhsa.fc.weight" => ("attn.out.weight", Linear),
            "mhsa.linear_pos.weight" => ("attn.pos.weight", Linear),
            "mhsa.pos_bias_u" => ("attn.pos_bias_u", PosBiasFlatten),
            "mhsa.pos_bias_v" => ("attn.pos_bias_v", PosBiasFlatten),
            "conv.pre_layer_norm.weight" => ("conv.norm.weight", F32Vector),
            "conv.pre_layer_norm.bias" => ("conv.norm.bias", F32Vector),
            "conv.pointwise_conv1.weight" => ("conv.pw1.weight", PointwiseConvSqueeze),
            "conv.depthwise_conv.weight" => ("conv.dw.weight", ConvKernel),
            // Upstream names this `batch_norm`, but it is an nn.LayerNorm over
            // the 2*d_model conv channels (no running stats exist).
            "conv.batch_norm.weight" => ("conv.ln.weight", F32Vector),
            "conv.batch_norm.bias" => ("conv.ln.bias", F32Vector),
            "conv.pointwise_conv2.weight" => ("conv.pw2.weight", PointwiseConvSqueeze),
            "layer_norm.weight" => ("out_norm.weight", F32Vector),
            "layer_norm.bias" => ("out_norm.bias", F32Vector),
            _ => return None,
        };
        return Some((format!("enc.blk.{layer}.{suffix}"), class));
    }
    if let Some(rest) = source_name.strip_prefix("decoder.layer_stack.") {
        let (layer, tail) = rest.split_once('.')?;
        let layer: usize = layer.parse().ok()?;
        let (suffix, class) = match tail {
            "self_attn_norm.weight" => ("self_attn.norm.weight", F32Vector),
            "self_attn_norm.bias" => ("self_attn.norm.bias", F32Vector),
            "self_attn.w_qs.weight" => ("self_attn.q.weight", Linear),
            "self_attn.w_qs.bias" => ("self_attn.q.bias", F32Vector),
            // w_ks is bias-free upstream (nn.Linear(..., bias=False)).
            "self_attn.w_ks.weight" => ("self_attn.k.weight", Linear),
            "self_attn.w_vs.weight" => ("self_attn.v.weight", Linear),
            "self_attn.w_vs.bias" => ("self_attn.v.bias", F32Vector),
            "self_attn.fc.weight" => ("self_attn.out.weight", Linear),
            "self_attn.fc.bias" => ("self_attn.out.bias", F32Vector),
            "cross_attn_norm.weight" => ("cross_attn.norm.weight", F32Vector),
            "cross_attn_norm.bias" => ("cross_attn.norm.bias", F32Vector),
            "cross_attn.w_qs.weight" => ("cross_attn.q.weight", Linear),
            "cross_attn.w_qs.bias" => ("cross_attn.q.bias", F32Vector),
            "cross_attn.w_ks.weight" => ("cross_attn.k.weight", Linear),
            "cross_attn.w_vs.weight" => ("cross_attn.v.weight", Linear),
            "cross_attn.w_vs.bias" => ("cross_attn.v.bias", F32Vector),
            "cross_attn.fc.weight" => ("cross_attn.out.weight", Linear),
            "cross_attn.fc.bias" => ("cross_attn.out.bias", F32Vector),
            "mlp_norm.weight" => ("ffn.norm.weight", F32Vector),
            "mlp_norm.bias" => ("ffn.norm.bias", F32Vector),
            "mlp.w_1.weight" => ("ffn.up.weight", Linear),
            "mlp.w_1.bias" => ("ffn.up.bias", F32Vector),
            "mlp.w_2.weight" => ("ffn.down.weight", Linear),
            "mlp.w_2.bias" => ("ffn.down.bias", F32Vector),
            _ => return None,
        };
        return Some((format!("dec.blk.{layer}.{suffix}"), class));
    }
    None
}

/// Target dims for one mapped tensor. Rank>=2 dims are reversed (torch
/// `[out, in]` -> ggml `[in, out]` for `mul_mat`; conv kernels land on the
/// ggml im2col layout), pointwise conv kernels squeeze their trailing k=1
/// axis first, and the rel-pos bias flattens to 1-D.
fn target_dims_for_class(
    source_shape: &[u64],
    class: TensorClass,
) -> Result<Vec<u64>, LocalSourceImportError> {
    match class {
        TensorClass::F32Vector => {
            if source_shape.len() != 1 {
                return Err(validate_error(format!(
                    "firered-aed f32-vector tensor must be rank 1, got {source_shape:?}"
                )));
            }
            Ok(source_shape.to_vec())
        }
        TensorClass::PosBiasFlatten => {
            if source_shape.len() != 2 {
                return Err(validate_error(format!(
                    "firered-aed rel-pos bias must be rank 2, got {source_shape:?}"
                )));
            }
            Ok(vec![source_shape[0] * source_shape[1]])
        }
        TensorClass::Linear => {
            if source_shape.len() != 2 {
                return Err(validate_error(format!(
                    "firered-aed linear weight must be rank 2, got {source_shape:?}"
                )));
            }
            Ok(vec![source_shape[1], source_shape[0]])
        }
        TensorClass::PointwiseConvSqueeze => match source_shape {
            [out, input, 1] => Ok(vec![*input, *out]),
            _ => Err(validate_error(format!(
                "firered-aed pointwise conv must be [out, in, 1], got {source_shape:?}"
            ))),
        },
        TensorClass::ConvKernel | TensorClass::F16Table => {
            if source_shape.len() < 2 {
                return Err(validate_error(format!(
                    "firered-aed conv/table tensor must be rank >= 2, got {source_shape:?}"
                )));
            }
            let mut dims = source_shape.to_vec();
            dims.reverse();
            Ok(dims)
        }
    }
}

fn quantized_tensor_type_for_firered_tensor(
    class: TensorClass,
    dims: &[u64],
    quantization: FireRedAedQuantizationMode,
) -> Option<GgufWriteTensorType> {
    if quantization == FireRedAedQuantizationMode::Fp16 {
        return None;
    }
    if !matches!(
        class,
        TensorClass::Linear | TensorClass::PointwiseConvSqueeze
    ) {
        return None;
    }
    let ne0 = dims.first().copied()?;
    if !ne0.is_multiple_of(32_u64) {
        return None;
    }
    if quantization == FireRedAedQuantizationMode::Q4_K && ne0.is_multiple_of(256_u64) {
        return Some(GgufWriteTensorType::Q4_K);
    }
    Some(GgufWriteTensorType::Q8_0)
}

fn build_firered_runtime_tensors(
    safetensors: &SafetensorsFile,
    quantization: FireRedAedQuantizationMode,
) -> Result<Vec<GgufWriteTensor>, LocalSourceImportError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for tensor in &safetensors.header().tensors {
        // Fail closed on ANY unrecognized tensor: the AED checkpoint has no
        // droppable tensors, so an unmapped name means upstream drift this
        // importer has not been audited against.
        let Some((target_name, class)) = map_firered_tensor_name(tensor.name.as_str()) else {
            return Err(validate_error(format!(
                "firered-aed source has an unrecognized tensor '{}'",
                tensor.name
            )));
        };
        if !seen.insert(target_name.clone()) {
            return Err(validate_error(format!(
                "firered-aed import mapped duplicate destination tensor '{target_name}'"
            )));
        }
        let target_dims = target_dims_for_class(tensor.shape.as_slice(), class)?;
        let data = safetensors.tensor_data(tensor)?;
        let write_tensor = match quantized_tensor_type_for_firered_tensor(
            class,
            &target_dims,
            quantization,
        ) {
            Some(qtype) => {
                let values = decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
                let quantized = quantize_f32_to_ggml_tensor_data(qtype, &target_dims, &values)
                    .map_err(|error| {
                        validate_error(format!(
                            "firered-aed quantization failed for '{}' -> '{target_name}' ({qtype:?}): {error}",
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
            None => match class {
                TensorClass::F32Vector | TensorClass::PosBiasFlatten => {
                    let values =
                        decode_safetensors_payload_as_f32(&tensor.name, &tensor.dtype, data)?;
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
                _ => {
                    let bits =
                        decode_safetensors_payload_as_f16_bits(&tensor.name, &tensor.dtype, data)?;
                    GgufWriteTensor {
                        name: target_name,
                        dims: target_dims,
                        tensor_type: GgufWriteTensorType::F16,
                        data: encode_f16_bits_le(bits),
                    }
                }
            },
        };
        out.push(write_tensor);
    }
    if out.is_empty() {
        return Err(validate_error(
            "firered-aed import found no tensors in the source",
        ));
    }
    Ok(out)
}

/// Kaldi/HTK mel filterbank `[n_mels, fft_bins]` (peak-normalized triangles)
/// for the frontend phase, matching the dolphin fbank convention (both
/// frontends are kaldi-style 80-mel/25ms/10ms/16kHz).
fn build_mel_filterbank_tensor(n_mels: usize) -> GgufWriteTensor {
    let fft_bins = FFT_SIZE / 2 + 1;
    let high_hz = (SAMPLE_RATE_HZ as f32) / 2.0;
    let mel_low = hz_to_mel(MEL_LOW_HZ);
    let mel_high = hz_to_mel(high_hz);
    let mel_delta = (mel_high - mel_low) / (n_mels as f32 + 1.0);

    let mut filters = vec![0.0_f32; n_mels * fft_bins];
    for mel_idx in 0..n_mels {
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
        name: "firered.mel_filters".to_string(),
        dims: vec![n_mels as u64, fft_bins as u64],
        tensor_type: GgufWriteTensorType::F32,
        data: bytes,
    }
}

/// Kaldi/HTK mel scale: `mel(f) = 1127 * ln(1 + f / 700)`.
fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn firered_runtime_gguf_metadata(
    hparams: &FireRedAedDerivedHparams,
    request: &FireRedAedImportRequest,
    vocab_tokens: &[String],
) -> BTreeMap<String, GgufWriteValue> {
    let mut metadata = BTreeMap::new();
    let mut put_str = |key: &str, value: &str| {
        metadata.insert(key.to_string(), GgufWriteValue::String(value.to_string()));
    };
    put_str("general.architecture", FIRERED_AED_GGML_ARCHITECTURE_ID);
    put_str(OASR_METADATA_KEY_PACKAGE_VERSION, OASR_PACKAGE_VERSION_V1);
    put_str(OASR_METADATA_KEY_MODEL_FAMILY, FIRERED_AED_MODEL_FAMILY);
    put_str(
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        FIRERED_AED_GGML_ARCHITECTURE_ID,
    );
    put_str(
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        FIRERED_AED_AUDIO_FRONTEND_ID,
    );
    put_str(
        OASR_METADATA_KEY_DECODE_POLICY,
        FIRERED_AED_DECODE_POLICY_ID,
    );
    put_str(GGML_TOKENIZER_ID_KEY, FIRERED_AED_TOKENIZER_ID);
    put_str("openasr.model.id", &request.model_id);

    let mut put_u32 = |key: &str, value: u32| {
        metadata.insert(key.to_string(), GgufWriteValue::U32(value));
    };
    put_u32("firered.encoder.n_layers", hparams.encoder_n_layers as u32);
    put_u32("firered.encoder.d_model", hparams.d_model as u32);
    put_u32("firered.encoder.n_heads", hparams.n_heads as u32);
    put_u32("firered.encoder.head_dim", hparams.head_dim as u32);
    put_u32("firered.encoder.ffn_dim", hparams.encoder_ffn_dim as u32);
    put_u32("firered.encoder.conv_kernel", hparams.conv_kernel as u32);
    put_u32(
        "firered.encoder.subsample_channels",
        hparams.subsample_channels as u32,
    );
    put_u32(
        "firered.encoder.subsample_out_dim",
        hparams.subsample_out_dim as u32,
    );
    put_u32("firered.encoder.feature_dim", hparams.feature_dim as u32);
    put_u32("firered.encoder.pe_len", hparams.encoder_pe_len as u32);
    put_u32("firered.decoder.n_layers", hparams.decoder_n_layers as u32);
    put_u32("firered.decoder.ffn_dim", hparams.decoder_ffn_dim as u32);
    put_u32("firered.decoder.pe_len", hparams.decoder_pe_len as u32);
    put_u32("firered.vocab_size", hparams.vocab_size as u32);
    put_u32("firered.sos_token_id", hparams.sos_token_id);
    put_u32("firered.eos_token_id", hparams.eos_token_id);
    put_u32("firered.pad_token_id", hparams.pad_token_id);
    // fbank frontend contract.
    put_u32("firered.audio.sample_rate", SAMPLE_RATE_HZ);
    put_u32("firered.audio.n_fft", FFT_SIZE as u32);
    put_u32("firered.audio.frame_length_ms", FRAME_LENGTH_MS);
    put_u32("firered.audio.frame_shift_ms", FRAME_SHIFT_MS);
    put_u32("firered.audio.n_mels", hparams.feature_dim as u32);

    metadata.insert(
        "tokenizer.ggml.tokens".to_string(),
        GgufWriteValue::StringArray(vocab_tokens.to_vec()),
    );
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use TensorClass::*;

    #[test]
    fn maps_encoder_block_tensors() {
        assert_eq!(
            map_firered_tensor_name("encoder.layer_stack.0.mhsa.w_qs.weight"),
            Some(("enc.blk.0.attn.q.weight".to_string(), Linear))
        );
        assert_eq!(
            map_firered_tensor_name("encoder.layer_stack.15.ffn2.net.4.weight"),
            Some(("enc.blk.15.ffn2.down.weight".to_string(), Linear))
        );
        assert_eq!(
            map_firered_tensor_name("encoder.layer_stack.3.mhsa.pos_bias_u"),
            Some(("enc.blk.3.attn.pos_bias_u".to_string(), PosBiasFlatten))
        );
        assert_eq!(
            map_firered_tensor_name("encoder.layer_stack.7.conv.batch_norm.weight"),
            Some(("enc.blk.7.conv.ln.weight".to_string(), F32Vector))
        );
        assert_eq!(
            map_firered_tensor_name("encoder.layer_stack.7.conv.pointwise_conv1.weight"),
            Some((
                "enc.blk.7.conv.pw1.weight".to_string(),
                PointwiseConvSqueeze
            ))
        );
        assert_eq!(
            map_firered_tensor_name("encoder.layer_stack.7.conv.depthwise_conv.weight"),
            Some(("enc.blk.7.conv.dw.weight".to_string(), ConvKernel))
        );
    }

    #[test]
    fn maps_decoder_block_and_top_level_tensors() {
        assert_eq!(
            map_firered_tensor_name("decoder.layer_stack.0.self_attn.w_ks.weight"),
            Some(("dec.blk.0.self_attn.k.weight".to_string(), Linear))
        );
        assert_eq!(
            map_firered_tensor_name("decoder.layer_stack.15.cross_attn.fc.bias"),
            Some(("dec.blk.15.cross_attn.out.bias".to_string(), F32Vector))
        );
        assert_eq!(
            map_firered_tensor_name("decoder.tgt_word_emb.weight"),
            Some(("dec.tok_emb.weight".to_string(), F16Table))
        );
        assert_eq!(
            map_firered_tensor_name("decoder.tgt_word_prj.weight"),
            Some(("dec.out_proj.weight".to_string(), Linear))
        );
        assert_eq!(
            map_firered_tensor_name("encoder.input_preprocessor.conv.0.weight"),
            Some(("enc.subsample.conv1.weight".to_string(), ConvKernel))
        );
        // A self-attn key bias does not exist upstream; the name must be
        // rejected (fail-closed) rather than silently mapped.
        assert_eq!(
            map_firered_tensor_name("decoder.layer_stack.0.self_attn.w_ks.bias"),
            None
        );
        assert_eq!(map_firered_tensor_name("some.unknown.tensor"), None);
    }

    #[test]
    fn target_dims_reverse_squeeze_and_flatten() {
        assert_eq!(
            target_dims_for_class(&[5120, 1280], Linear).unwrap(),
            vec![1280, 5120]
        );
        assert_eq!(
            target_dims_for_class(&[5120, 1280, 1], PointwiseConvSqueeze).unwrap(),
            vec![1280, 5120]
        );
        assert_eq!(
            target_dims_for_class(&[2560, 1, 33], ConvKernel).unwrap(),
            vec![33, 1, 2560]
        );
        assert_eq!(
            target_dims_for_class(&[32, 1, 3, 3], ConvKernel).unwrap(),
            vec![3, 3, 1, 32]
        );
        assert_eq!(
            target_dims_for_class(&[20, 64], PosBiasFlatten).unwrap(),
            vec![1280]
        );
        assert_eq!(
            target_dims_for_class(&[1, 9999, 1280], F16Table).unwrap(),
            vec![1280, 9999, 1]
        );
        assert!(target_dims_for_class(&[5120, 1280, 3], PointwiseConvSqueeze).is_err());
        assert!(target_dims_for_class(&[1280, 5120], F32Vector).is_err());
    }

    #[test]
    fn quantizes_only_linear_classes_with_aligned_ne0() {
        assert_eq!(
            quantized_tensor_type_for_firered_tensor(
                Linear,
                &[1280, 5120],
                FireRedAedQuantizationMode::Q4_K
            ),
            Some(GgufWriteTensorType::Q4_K)
        );
        assert_eq!(
            quantized_tensor_type_for_firered_tensor(
                PointwiseConvSqueeze,
                &[1280, 5120],
                FireRedAedQuantizationMode::Q8_0
            ),
            Some(GgufWriteTensorType::Q8_0)
        );
        // fp16 mode never quantizes; conv kernels / tables / vectors never do.
        assert_eq!(
            quantized_tensor_type_for_firered_tensor(
                Linear,
                &[1280, 5120],
                FireRedAedQuantizationMode::Fp16
            ),
            None
        );
        assert_eq!(
            quantized_tensor_type_for_firered_tensor(
                ConvKernel,
                &[33, 1, 2560],
                FireRedAedQuantizationMode::Q4_K
            ),
            None
        );
        assert_eq!(
            quantized_tensor_type_for_firered_tensor(
                F16Table,
                &[1280, 7832, 1],
                FireRedAedQuantizationMode::Q4_K
            ),
            None
        );
        // Unaligned ne0 falls back to the f16 representation.
        assert_eq!(
            quantized_tensor_type_for_firered_tensor(
                Linear,
                &[100, 5120],
                FireRedAedQuantizationMode::Q8_0
            ),
            None
        );
    }

    #[test]
    fn parses_kaldi_cmvn_stats_with_upstream_formula() {
        // dim=2, count=4: sums [8, 4], sumsq [32, 8].
        // mean = [2, 1]; var = [32/4 - 4, 8/4 - 1] = [4, 1]; istd = [0.5, 1].
        let dir = std::env::temp_dir().join(format!("firered-cmvn-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cmvn.txt");
        std::fs::write(&path, " [\n  8.0 4.0 4.0 \n  32.0 8.0 0.0 ]\n").unwrap();
        let (neg_mean, inv_stddev) = parse_kaldi_cmvn_stats(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(neg_mean, vec![-2.0, -1.0]);
        assert_eq!(inv_stddev, vec![0.5, 1.0]);
    }

    #[test]
    fn dict_txt_requires_contiguous_ids_and_maps_space() {
        let dir = std::env::temp_dir().join(format!("firered-dict-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dict.txt");
        std::fs::write(&path, "<blank> 0\n<unk> 1\n<space> 2\n\u{2581}A 3\n").unwrap();
        let tokens = read_dict_txt(&path).unwrap();
        assert_eq!(tokens, vec!["<blank>", "<unk>", " ", "\u{2581}A"]);

        std::fs::write(&path, "<blank> 0\n<unk> 2\n").unwrap();
        assert!(read_dict_txt(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn metadata_declares_family_and_contract_keys() {
        let hparams = FireRedAedDerivedHparams {
            encoder_n_layers: 16,
            decoder_n_layers: 16,
            d_model: 1280,
            n_heads: 20,
            head_dim: 64,
            encoder_ffn_dim: 5120,
            decoder_ffn_dim: 5120,
            conv_kernel: 33,
            subsample_channels: 32,
            subsample_out_dim: 608,
            feature_dim: 80,
            encoder_pe_len: 9999,
            decoder_pe_len: 5000,
            vocab_size: 7832,
            sos_token_id: 3,
            eos_token_id: 4,
            pad_token_id: 2,
        };
        let request = FireRedAedImportRequest {
            source_root: PathBuf::from("/tmp/firered"),
            output_root: PathBuf::from("/tmp/firered.oasr"),
            model_id: "firered-aed-l".to_string(),
            quantization: FireRedAedQuantizationMode::Fp16,
        };
        let tokens: Vec<String> = (0..7832).map(|i| format!("t{i}")).collect();
        let metadata = firered_runtime_gguf_metadata(&hparams, &request, &tokens);
        assert!(matches!(
            metadata.get(OASR_METADATA_KEY_MODEL_FAMILY),
            Some(GgufWriteValue::String(family)) if family == FIRERED_AED_MODEL_FAMILY
        ));
        assert!(matches!(
            metadata.get("firered.encoder.d_model"),
            Some(GgufWriteValue::U32(1280))
        ));
        assert!(matches!(
            metadata.get("firered.sos_token_id"),
            Some(GgufWriteValue::U32(3))
        ));
        assert!(matches!(
            metadata.get("tokenizer.ggml.tokens"),
            Some(GgufWriteValue::StringArray(list)) if list.len() == 7832
        ));
    }
}
