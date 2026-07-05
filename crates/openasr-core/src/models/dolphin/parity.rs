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

use crate::ggml_runtime::GgmlCpuGraphBackend;

use super::decoder_graph::{DolphinDecoderConfig, decode_prompt_logits};
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

/// Like [`load_safetensors_f32`] but only materializes tensors whose name starts
/// with `prefix`. `full.safetensors` carries the whole 1.7 GB state dict; the
/// decoder harness only needs the `decoder.*` namespace, so filtering keeps the
/// working set small.
fn load_safetensors_f32_prefixed(path: &Path, prefix: &str) -> HashMap<String, Vec<f32>> {
    let bytes = std::fs::read(path).expect("read safetensors");
    assert!(bytes.len() >= 8, "safetensors too short");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..header_end]).expect("parse safetensors header");
    let obj = header.as_object().expect("header object");

    let mut out = HashMap::new();
    for (name, meta) in obj {
        if name == "__metadata__" || !name.starts_with(prefix) {
            continue;
        }
        let dtype = meta["dtype"].as_str().expect("dtype");
        assert_eq!(dtype, "F32", "expected f32 weights, got {dtype} for {name}");
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

    // Parity is CPU-locked: the golden fixtures are bit-exact against the CPU
    // graph; the backend param exists so the runtime can pick Metal, not the gate.
    let output = encode(
        &config,
        &weights,
        &features,
        frames_in,
        GgmlCpuGraphBackend::Cpu,
    )
    .expect("encode");

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

/// Canonical Dolphin decode prefix (OWSM-style): sos + lang + region + task +
/// timestamp = `<sos><zh><SICHUAN><asr><notimestamp>`. The golden
/// `decoder_step0_logits` is the distribution predicting the first content token
/// given this whole prompt (the last row of the teacher-forced prompt logits).
const DECODER_PROMPT: [u32; 5] = [2, 5, 10, 4, 109];

#[test]
#[ignore = "requires local Dolphin full.safetensors + golden under tmp/publish (not committed)"]
fn dolphin_decoder_parity() {
    let root = root();
    let weights_path = root.join("weights/full.safetensors");
    let encoder_out_path = root.join("golden/encoder_out.npy");
    let step0_path = root.join("golden/decoder_step0_logits.npy");
    if !weights_path.exists() || !step0_path.exists() {
        eprintln!("skip: dolphin decoder weights/golden not present under {root:?}");
        return;
    }

    let weights = load_safetensors_f32_prefixed(&weights_path, "decoder.");
    let (enc_shape, encoder_out) = load_npy_f32(&encoder_out_path);
    assert_eq!(enc_shape.len(), 3, "expected (1,T',768), got {enc_shape:?}");
    let frames = enc_shape[1];
    let d_model = enc_shape[2];

    let config = DolphinDecoderConfig::small_cn();
    assert_eq!(d_model, config.d_model, "encoder hidden mismatch");

    let output = decode_prompt_logits(
        &config,
        &weights,
        &encoder_out,
        frames,
        &DECODER_PROMPT,
        GgmlCpuGraphBackend::Cpu,
    )
    .expect("dolphin decoder");
    assert_eq!(output.token_count, DECODER_PROMPT.len());
    assert_eq!(output.vocab_size, config.vocab_size);

    let actual = output.last_token_logits();
    let (gshape, golden) = load_npy_f32(&step0_path);
    assert_eq!(gshape, vec![config.vocab_size], "golden step0 shape");

    let (max, mean) = diff(actual, &golden);
    let rel = relative_max_diff(actual, &golden);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    };
    println!("== Dolphin Transformer decoder parity (first content step) ==");
    println!(
        "prompt {DECODER_PROMPT:?} -> {frames} encoder frames, {} tokens",
        output.token_count
    );
    println!(
        "decoder_step0 : max {max:.3e}  mean {mean:.3e}  rel {rel:.3e}  \
         argmax actual={} golden={} (expect 3805)",
        argmax(actual),
        argmax(&golden)
    );

    // The decoder graph is assembled entirely in f32 (attention via
    // mul_mat/soft_max_ext/mul_mat, no f16 KV cache), so it reproduces the
    // PyTorch reference down to the f32 accumulation-order noise floor. Gate at
    // the task's 1e-3 bound; an algorithmic divergence (wrong scale, mask,
    // norm placement, cross-attn source) blows this up by orders of magnitude.
    assert_eq!(
        argmax(actual),
        argmax(&golden),
        "decoder first content token mismatch"
    );
    assert!(
        max < 1.0e-3,
        "decoder_step0 max abs diff {max:.3e} exceeds the 1e-3 parity bound"
    );
}
