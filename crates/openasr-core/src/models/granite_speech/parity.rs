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

use super::encoder_graph::{GraniteSpeechEncoderConfig, encode};
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
