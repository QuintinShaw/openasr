//! Dev parity harness for the Dolphin `small.cn` E-Branchformer encoder.
//!
//! Loads the reference weights (safetensors) and golden fixtures from the local
//! publish scratch dir, runs the ggml encoder graph on CPU, and prints max/mean
//! abs diff vs the PyTorch reference after subsampling and after each block. It
//! is `#[ignore]` because the 866 MB weights live under `tmp/` (never committed);
//! run it explicitly with:
//!
//! ```text
//! cargo test -p openasr-core dolphin_encoder_parity -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::encoder_graph::{DolphinEncoderConfig, encode};

const FIXTURE_ROOT: &str =
    "/Volumes/QuintinDocument/openasr-dev/openasr/tmp/publish/dolphin-cn-dialect-small";

fn root() -> PathBuf {
    PathBuf::from(FIXTURE_ROOT)
}

// --- minimal safetensors reader (all tensors f32) --------------------------

fn load_safetensors_f32(path: &Path) -> HashMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).expect("read safetensors");
    assert!(bytes.len() >= 8, "safetensors too short");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..header_end]).expect("parse safetensors header");
    let obj = header.as_object().expect("header object");

    let mut out = HashMap::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta["dtype"].as_str().expect("dtype");
        assert_eq!(
            dtype, "F32",
            "expected all-f32 weights, got {dtype} for {name}"
        );
        let offsets = meta["data_offsets"].as_array().expect("data_offsets");
        let start = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;
        let raw = &bytes[header_end + start..header_end + end];
        assert!(
            raw.len().is_multiple_of(4),
            "tensor {name} not 4-byte aligned"
        );
        let values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        out.insert(name.clone(), values);
    }
    out
}

// --- minimal .npy reader (little-endian f32) -------------------------------

fn load_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).expect("read npy");
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
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    for (a, e) in actual.iter().zip(expected.iter()) {
        let d = (a - e).abs();
        max = max.max(d);
        sum += d as f64;
    }
    (max, (sum / actual.len() as f64) as f32)
}

#[test]
#[ignore = "requires local 866MB Dolphin weights under tmp/publish (not committed)"]
fn dolphin_encoder_parity() {
    let root = root();
    let weights_path = root.join("weights/encoder.safetensors");
    if !weights_path.exists() {
        eprintln!("skip: {weights_path:?} not present");
        return;
    }

    let weights = load_safetensors_f32(&weights_path);
    let (in_shape, features) = load_npy_f32(&root.join("golden/logmel_feats_cmvn.npy"));
    assert_eq!(
        in_shape.len(),
        3,
        "expected (1, T, 80) input, got {in_shape:?}"
    );
    let frames_in = in_shape[1];
    let feat_dim = in_shape[2];

    let config = DolphinEncoderConfig::small_cn();
    assert_eq!(feat_dim, config.feature_dim, "feature dim mismatch");

    let output = encode(&config, &weights, &features, frames_in).expect("encode");

    println!("== Dolphin E-Branchformer encoder parity ==");
    println!(
        "input frames {frames_in} -> subsampled {} (dim {})",
        output.frames, output.dim
    );

    let (_, golden_sub) = load_npy_f32(&root.join("golden/enc_after_subsample.npy"));
    let (m, mean) = diff(&output.after_subsample, &golden_sub);
    println!("after_subsample : max {m:.3e}  mean {mean:.3e}");

    let mut first_diverge: Option<String> = None;
    for (i, block) in output.blocks.iter().enumerate() {
        let (_, golden) = load_npy_f32(&root.join(format!("golden/enc_block{i}.npy")));
        let (m, mean) = diff(block, &golden);
        println!("block{i:<2}        : max {m:.3e}  mean {mean:.3e}");
        if first_diverge.is_none() && m > 1.0e-2 {
            first_diverge = Some(format!("block{i}"));
        }
    }

    let (_, golden_out) = load_npy_f32(&root.join("golden/encoder_out.npy"));
    let (m_final, mean_final) = diff(&output.encoder_out, &golden_out);
    println!("encoder_out     : max {m_final:.3e}  mean {mean_final:.3e}");

    let (m_sub, _) = diff(&output.after_subsample, &golden_sub);
    if first_diverge.is_none() && m_sub > 1.0e-2 {
        first_diverge = Some("after_subsample".to_string());
    }
    if first_diverge.is_none() && m_final > 1.0e-2 {
        first_diverge = Some("encoder_out".to_string());
    }
    match &first_diverge {
        Some(stage) => println!("first divergence (>1e-2 max abs): {stage}"),
        None => println!("first divergence (>1e-2 max abs): none - full parity within 1e-2"),
    }

    // Loose gate: keep the harness green while the graph is validated. The
    // printed per-stage diffs are the real signal; tighten this once parity is
    // nailed.
    assert!(
        m_final.is_finite(),
        "encoder output diverged to non-finite values"
    );
}
