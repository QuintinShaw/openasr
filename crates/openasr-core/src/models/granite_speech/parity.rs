//! Dev parity harness for the Granite Speech 4.1 encoder + Q-Former projector.
//!
//! Loads the reference weights (bf16 safetensors, upcast to f32) and golden
//! fixtures dumped from HF `transformers` (fp32 forward), runs the ggml graph
//! on CPU, and asserts max-abs-diff tolerance. `#[ignore]`: the 4.6 GB
//! checkpoint and fixtures live under `tmp/` (never committed). Run with:
//!
//! ```text
//! cargo test -p openasr-core granite_speech_encoder_parity -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ggml_runtime::GgmlCpuGraphBackend;

use super::decoder_graph::{GraniteSpeechDecoderConfig, prefill_logits};
use super::encoder_graph::{GraniteSpeechEncoderConfig, encode};
use super::frontend::GraniteSpeechMelFrontend;
use super::qformer::{GraniteSpeechProjectorConfig, project};

const WEIGHTS_ROOT: &str =
    "/Volumes/QuintinDocument/openasr-dev/tmp/granite-work/granite-speech-4.1-2b-src";
const GOLDEN_ROOT: &str = "/Volumes/QuintinDocument/openasr-dev/tmp/granite-work/golden";

fn weights_root() -> PathBuf {
    PathBuf::from(WEIGHTS_ROOT)
}

fn golden_root() -> PathBuf {
    PathBuf::from(GOLDEN_ROOT)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Loads every tensor whose name starts with `prefix` from a (possibly
/// sharded) safetensors checkpoint, upcasting BF16 to F32 (this checkpoint's
/// native dtype; see `config.json`'s `"dtype": "bfloat16"`). F32 tensors pass
/// through unchanged.
fn load_safetensors_prefixed(dir: &Path, prefix: &str) -> HashMap<String, Vec<f32>> {
    let index_path = dir.join("model.safetensors.index.json");
    let index_bytes = std::fs::read(&index_path).expect("read safetensors index");
    let index: serde_json::Value = serde_json::from_slice(&index_bytes).expect("parse index");
    let weight_map = index["weight_map"].as_object().expect("weight_map object");

    let mut shard_names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    shard_names.sort();
    shard_names.dedup();

    let mut out = HashMap::new();
    for shard in shard_names {
        let path = dir.join(&shard);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert!(bytes.len() >= 8, "safetensors too short");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..header_end]).expect("parse safetensors header");
        let obj = header.as_object().expect("header object");
        for (name, meta) in obj {
            if name == "__metadata__" || !name.starts_with(prefix) {
                continue;
            }
            let dtype = meta["dtype"].as_str().expect("dtype");
            let offsets = meta["data_offsets"].as_array().expect("data_offsets");
            let start = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;
            let raw = &bytes[header_end + start..header_end + end];
            let values: Vec<f32> = match dtype {
                "F32" => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                "BF16" => raw
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                // Bookkeeping scalars (e.g. BatchNorm's `num_batches_tracked`,
                // an I64 training-step counter) carry no forward-pass value;
                // skip anything we don't need instead of failing the load.
                _ => continue,
            };
            out.insert(name.clone(), values);
        }
    }
    out
}

fn load_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
    let major = bytes[6];
    let header_len = if major == 1 {
        u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
    } else {
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .expect("npy header utf8");
    assert!(header.contains("'<f4'"), "expected <f4 npy, got {header}");
    assert!(
        header.contains("'fortran_order': False"),
        "expected C order"
    );

    let shape_start = header.find("'shape':").expect("shape key");
    let paren = header[shape_start..].find('(').unwrap() + shape_start;
    let close = header[paren..].find(')').unwrap() + paren;
    let shape: Vec<usize> = header[paren + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    let data_start = header_start + header_len;
    let values: Vec<f32> = bytes[data_start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (shape, values)
}

fn diff(actual: &[f32], expected: &[f32]) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let d = (a - e).abs();
        max = max.max(d);
        sum += d as f64;
    }
    (max, (sum / actual.len() as f64) as f32)
}

fn relative_max_diff(actual: &[f32], expected: &[f32]) -> f32 {
    let (max, _) = diff(actual, expected);
    let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if scale > 0.0 { max / scale } else { max }
}

#[test]
#[ignore = "requires local 4.6GB granite-speech-4.1-2b weights + golden fixtures under tmp/ (not committed)"]
fn granite_speech_encoder_parity() {
    let weights_dir = weights_root();
    if !weights_dir.join("model.safetensors.index.json").exists() {
        eprintln!("skip: {weights_dir:?} not present");
        return;
    }
    let golden = golden_root();

    let weights = load_safetensors_prefixed(&weights_dir, "encoder.");
    let (in_shape, features) = load_npy_f32(&golden.join("en_short_input_features.npy"));
    assert_eq!(in_shape.len(), 3, "expected (1,T,160), got {in_shape:?}");
    let frames = in_shape[1];
    let input_dim = in_shape[2];

    let config = GraniteSpeechEncoderConfig::granite_speech_4_1_2b();
    assert_eq!(input_dim, config.input_dim, "input dim mismatch");

    let output = encode(
        &config,
        &weights,
        &features,
        frames,
        GgmlCpuGraphBackend::Cpu,
        true,
    )
    .expect("encode");

    println!("== Granite Speech encoder parity ==");
    println!("input frames {frames} -> encoder_out (dim {})", output.dim);

    let (mid_shape, golden_mid) = load_npy_f32(&golden.join("en_short_encoder_mid_block_out.npy"));
    assert_eq!(
        mid_shape,
        vec![1, frames, config.hidden_dim],
        "golden mid shape"
    );
    let (m_mid, mean_mid) = diff(&output.mid_block_out, &golden_mid);
    let rel_mid = relative_max_diff(&output.mid_block_out, &golden_mid);
    println!(
        "mid_block_out (post layer 8, pre-CTC-tap): max {m_mid:.3e}  mean {mean_mid:.3e}  rel {rel_mid:.3e}"
    );

    let (out_shape, golden_out) = load_npy_f32(&golden.join("en_short_encoder_out.npy"));
    assert_eq!(
        out_shape,
        vec![1, frames, config.hidden_dim],
        "golden encoder_out shape"
    );
    let (m_final, mean_final) = diff(&output.encoder_out, &golden_out);
    let rel_final = relative_max_diff(&output.encoder_out, &golden_out);
    println!("encoder_out: max {m_final:.3e}  mean {mean_final:.3e}  rel {rel_final:.3e}");

    assert!(
        m_mid < 1.0e-3,
        "mid_block_out max abs diff {m_mid:.3e} exceeds the 1e-3 parity bound"
    );
    assert!(
        m_final < 1.0e-3,
        "encoder_out max abs diff {m_final:.3e} exceeds the 1e-3 parity bound"
    );
}

#[test]
#[ignore = "requires local 4.6GB granite-speech-4.1-2b weights + golden fixtures under tmp/ (not committed)"]
fn granite_speech_projector_parity() {
    let weights_dir = weights_root();
    if !weights_dir.join("model.safetensors.index.json").exists() {
        eprintln!("skip: {weights_dir:?} not present");
        return;
    }
    let golden = golden_root();

    let encoder_weights = load_safetensors_prefixed(&weights_dir, "encoder.");
    let projector_weights = load_safetensors_prefixed(&weights_dir, "projector.");
    let (in_shape, features) = load_npy_f32(&golden.join("en_short_input_features.npy"));
    let frames = in_shape[1];

    let enc_config = GraniteSpeechEncoderConfig::granite_speech_4_1_2b();
    let encoder = encode(
        &enc_config,
        &encoder_weights,
        &features,
        frames,
        GgmlCpuGraphBackend::Cpu,
        false,
    )
    .expect("encode");

    let proj_config = GraniteSpeechProjectorConfig::granite_speech_4_1_2b();
    let output = project(
        &proj_config,
        &projector_weights,
        &encoder.encoder_out,
        encoder.frames,
        GgmlCpuGraphBackend::Cpu,
    )
    .expect("project");

    println!("== Granite Speech Q-Former projector parity ==");
    println!(
        "encoder frames {frames} -> projector tokens {}",
        output.tokens
    );

    let (out_shape, golden_out) = load_npy_f32(&golden.join("en_short_projector_out.npy"));
    assert_eq!(out_shape.len(), 3, "expected (1,N,2048), got {out_shape:?}");
    assert_eq!(out_shape[1], output.tokens, "token count mismatch");
    assert_eq!(out_shape[2], output.dim, "dim mismatch");

    let (m, mean) = diff(&output.projected, &golden_out);
    let rel = relative_max_diff(&output.projected, &golden_out);
    println!("projector_out: max {m:.3e}  mean {mean:.3e}  rel {rel:.3e}");

    assert!(
        m < 1.0e-2,
        "projector_out max abs diff {m:.3e} exceeds the 1e-2 parity bound"
    );
}

/// Little-endian `<i8` (int64) npy reader, for `decoder_input_ids.npy`.
fn load_npy_i64(path: &Path) -> (Vec<usize>, Vec<i64>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
    let major = bytes[6];
    let header_len = if major == 1 {
        u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
    } else {
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])
        .expect("npy header utf8");
    assert!(header.contains("'<i8'"), "expected <i8 npy, got {header}");
    assert!(
        header.contains("'fortran_order': False"),
        "expected C order"
    );

    let shape_start = header.find("'shape':").expect("shape key");
    let paren = header[shape_start..].find('(').unwrap() + shape_start;
    let close = header[paren..].find(')').unwrap() + paren;
    let shape: Vec<usize> = header[paren + 1..close]
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    let data_start = header_start + header_len;
    let values: Vec<i64> = bytes[data_start..]
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (shape, values)
}

fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());
    idx.truncate(k);
    idx
}

#[test]
#[ignore = "requires local 4.6GB granite-speech-4.1-2b weights + golden fixtures under tmp/ (not committed)"]
fn granite_speech_decoder_prefill_parity() {
    let weights_dir = weights_root();
    if !weights_dir.join("model.safetensors.index.json").exists() {
        eprintln!("skip: {weights_dir:?} not present");
        return;
    }
    let golden = golden_root();

    let weights = load_safetensors_prefixed(&weights_dir, "language_model.");
    let (ids_shape, ids_i64) = load_npy_i64(&golden.join("decoder_input_ids.npy"));
    assert_eq!(ids_shape.len(), 2, "expected (1,T), got {ids_shape:?}");
    let n_tokens = ids_shape[1];
    let token_ids: Vec<u32> = ids_i64.iter().map(|&id| id as u32).collect();

    let config = GraniteSpeechDecoderConfig::granite_speech_4_1_2b();
    let output = prefill_logits(&config, &weights, &token_ids, GgmlCpuGraphBackend::Cpu)
        .expect("prefill_logits");

    println!("== Granite Speech decoder prefill parity ==");
    println!(
        "n_tokens {n_tokens} -> logits [n_tokens, {}]",
        output.vocab_size
    );

    let (hidden_shape, golden_hidden) = load_npy_f32(&golden.join("decoder_hidden_out.npy"));
    assert_eq!(
        hidden_shape,
        vec![1, n_tokens, config.hidden_size],
        "golden hidden shape"
    );
    let (m_hidden, mean_hidden) = diff(&output.hidden_out, &golden_hidden);
    let rel_hidden = relative_max_diff(&output.hidden_out, &golden_hidden);
    println!("hidden_out: max {m_hidden:.3e}  mean {mean_hidden:.3e}  rel {rel_hidden:.3e}");

    let (logits_shape, golden_logits) = load_npy_f32(&golden.join("decoder_logits.npy"));
    assert_eq!(
        logits_shape,
        vec![1, n_tokens, config.vocab_size],
        "golden logits shape"
    );
    let (m_logits, mean_logits) = diff(&output.logits, &golden_logits);
    let rel_logits = relative_max_diff(&output.logits, &golden_logits);
    println!("logits:     max {m_logits:.3e}  mean {mean_logits:.3e}  rel {rel_logits:.3e}");

    // Last-position top-10 by argmax-set agreement (order-sensitive logit
    // differences at the ~1e-3 scale can swap near-tied ranks without
    // reflecting a real divergence; the set match is the honest gate here).
    let last_start = (n_tokens - 1) * config.vocab_size;
    let actual_last = &output.logits[last_start..last_start + config.vocab_size];
    let golden_last = &golden_logits[last_start..last_start + config.vocab_size];
    let actual_top10 = top_k_indices(actual_last, 10);
    let golden_top10 = top_k_indices(golden_last, 10);
    println!("last-position actual top10: {actual_top10:?}");
    println!("last-position golden top10: {golden_top10:?}");
    assert_eq!(
        actual_top10[0], golden_top10[0],
        "argmax token mismatch: actual {} vs golden {}",
        actual_top10[0], golden_top10[0]
    );
    let overlap = actual_top10
        .iter()
        .filter(|id| golden_top10.contains(id))
        .count();
    assert!(
        overlap >= 9,
        "top10 overlap only {overlap}/10 (actual {actual_top10:?} vs golden {golden_top10:?})"
    );

    assert!(
        m_hidden < 1.0e-2,
        "hidden_out max abs diff {m_hidden:.3e} exceeds the 1e-2 parity bound"
    );
    assert!(
        m_logits < 5.0e-2,
        "logits max abs diff {m_logits:.3e} exceeds the 5e-2 parity bound"
    );
}

/// Minimal PCM16LE mono WAV reader (44-byte canonical header), for the
/// frontend parity sample only -- not a general-purpose loader (the crate's
/// `audio::prepare`/`symphonia_decode` own that job for the real pipeline).
fn load_wav_pcm16_mono_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(&bytes[0..4], b"RIFF", "expected RIFF wav");
    assert_eq!(&bytes[8..12], b"WAVE", "expected WAVE wav");
    // Walk chunks to find "data" (canonical 44-byte header assumed absent
    // any extra chunks, which is what ffmpeg's pcm_s16le writer emits).
    let mut cursor = 12usize;
    let mut data_offset = None;
    let mut data_len = 0usize;
    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_len =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        if chunk_id == b"data" {
            data_offset = Some(cursor + 8);
            data_len = chunk_len;
            break;
        }
        cursor += 8 + chunk_len + (chunk_len % 2);
    }
    let data_offset = data_offset.expect("wav data chunk");
    bytes[data_offset..data_offset + data_len]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32 / 32768.0)
        .collect()
}

#[test]
fn granite_speech_frontend_parity() {
    let golden = golden_root();
    let sample_path =
        PathBuf::from("/Volumes/QuintinDocument/openasr-dev/tmp/granite-work/samples/en_short.wav");
    let golden_path = golden.join("en_short_input_features.npy");
    if !sample_path.exists() || !golden_path.exists() {
        eprintln!("skip: {sample_path:?} or {golden_path:?} not present");
        return;
    }

    let samples = load_wav_pcm16_mono_f32(&sample_path);
    let frontend = GraniteSpeechMelFrontend::new();
    let (actual, frames) = frontend.extract(&samples).expect("extract");

    let (golden_shape, golden_features) = load_npy_f32(&golden_path);
    assert_eq!(
        golden_shape.len(),
        3,
        "expected (1,T,160), got {golden_shape:?}"
    );
    assert_eq!(frames, golden_shape[1], "frame count mismatch");
    assert_eq!(
        actual.len(),
        golden_features.len(),
        "element count mismatch"
    );

    let (m, mean) = diff(&actual, &golden_features);
    let rel = relative_max_diff(&actual, &golden_features);
    println!("== Granite Speech mel frontend parity ==");
    println!("frames {frames}: max {m:.3e}  mean {mean:.3e}  rel {rel:.3e}");

    assert!(
        m < 1.0e-2,
        "input_features max abs diff {m:.3e} exceeds the 1e-2 parity bound"
    );
}
