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

/// Scale-invariant max abs diff: `max|a-e| / max|e|`. Absolute diffs scale with
/// each tap's magnitude (the post-subsample hidden is ~80x larger than the final
/// output because of the `sqrt(d_model)` xscale and no output LayerNorm), so the
/// relative error is the honest cross-tap parity metric.
fn relative_max_diff(actual: &[f32], expected: &[f32]) -> f32 {
    let (max, _) = diff(actual, expected);
    let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if scale > 0.0 { max / scale } else { max }
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
    let (m_sub, mean_sub) = diff(&output.after_subsample, &golden_sub);
    let rel_sub = relative_max_diff(&output.after_subsample, &golden_sub);
    println!("after_subsample : max {m_sub:.3e}  mean {mean_sub:.3e}  rel {rel_sub:.3e}");

    let mut first_diverge: Option<String> = None;
    let mut worst_block: f32 = 0.0;
    for (i, block) in output.blocks.iter().enumerate() {
        let (_, golden) = load_npy_f32(&root.join(format!("golden/enc_block{i}.npy")));
        let (m, mean) = diff(block, &golden);
        let rel = relative_max_diff(block, &golden);
        println!("block{i:<2}        : max {m:.3e}  mean {mean:.3e}  rel {rel:.3e}");
        worst_block = worst_block.max(m);
        if first_diverge.is_none() && m > 1.0e-2 {
            first_diverge = Some(format!("block{i}"));
        }
    }

    let (_, golden_out) = load_npy_f32(&root.join("golden/encoder_out.npy"));
    let (m_final, mean_final) = diff(&output.encoder_out, &golden_out);
    let rel_final = relative_max_diff(&output.encoder_out, &golden_out);
    println!("encoder_out     : max {m_final:.3e}  mean {mean_final:.3e}  rel {rel_final:.3e}");

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

    // Parity gate. The E-Branchformer encoder is bit-exact with the PyTorch
    // reference down to the f32 accumulation-order noise floor: every tap sits at
    // ~1e-6 *relative* max diff (f32 eps is ~1.2e-7). Absolute diffs are gated at
    // the task's cumulative 1e-3 bit-exact bound (encoder_out lands ~1e-5, ~90x of
    // headroom for thread-order variation); after_subsample carries a larger
    // *absolute* diff only because its values are ~80x bigger (sqrt(d_model) xscale
    // with no output LayerNorm), so it is gated on the scale-invariant relative
    // error instead. If any of these trip, a stage genuinely diverged.
    assert!(
        first_diverge.is_none(),
        "encoder diverged from the reference (>1e-2 max abs) at {first_diverge:?}"
    );
    assert!(
        rel_sub < 1.0e-4,
        "after_subsample relative max diff {rel_sub:.3e} exceeds 1e-4 - algorithmic divergence, not f32 noise"
    );
    assert!(
        worst_block < 1.0e-3,
        "an encoder block max abs diff {worst_block:.3e} exceeds the 1e-3 parity bound"
    );
    assert!(
        m_final < 1.0e-3,
        "encoder_out max abs diff {m_final:.3e} exceeds the 1e-3 parity bound"
    );
}
