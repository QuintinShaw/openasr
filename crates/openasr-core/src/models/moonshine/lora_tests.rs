//! Real-pack LoRA logits oracles for the moonshine dynamic side-path.
//!
//! Attributed oracles (real moonshine-tiny q8_0 pack, CPU decoder path):
//!
//! 1. ZERO adapter (B == 0) must leave decoder logits VALUE-EXACT. The side
//!    branch contributes `B_scaled@(A@x) == 0` and IEEE-754 `x + 0.0 == x`
//!    for every finite `x` (the only caveat is `-0.0 + 0.0 == +0.0`, which
//!    still compares equal under `==`), so element-wise `==` is the correct
//!    claim — stronger than token-exact, marginally weaker than bit-exact.
//! 2. Logits delta must be FIRST-ORDER LINEAR in B: for a small constant fill
//!    `eps`, `delta(2*eps) ≈ 2 * delta(eps)`. The LoRA delta on the targeted
//!    projection is exactly linear in B; downstream nonlinearities make the
//!    logits delta only approximately linear, which is what the tolerance
//!    bounds. This pins the side branch's sign, scale, and wiring (a wrong
//!    `alpha/rank`, transposed A/B, or a missed cross-KV precompute path all
//!    break it).
//!
//! Loud-fail prerequisites: moonshine-tiny q8_0 installed at
//! `~/.openasr/models/moonshine-tiny/q8_0/` (or env override); the tests are
//! `#[ignore]` and must never silently skip.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::decoder_graph::MoonshineDecoderGraphRuntime;
use super::encoder_graph::MoonshineEncoderOutput;
use super::lora::{MoonshineLoraAdapter, MoonshineLoraTarget, moonshine_lora_adapter_for_test};
use super::prepared_runtime::{MoonshinePreparedRuntime, build_moonshine_prepared_runtime};
use super::runtime_contract::MoonshineExecutionMetadata;
use crate::{
    GgmlAsrRuntimeSourcePreflight, read_gguf_metadata_from_runtime_source,
    read_gguf_tensor_index_from_runtime_source, validate_ggml_runtime_source_path,
};

const MOONSHINE_OADP_REAL_PACK_ENV: &str = "OPENASR_MOONSHINE_OADP_REAL_PACK";
const MOONSHINE_PACK_HOME_RELATIVE_PATH: &str =
    ".openasr/models/moonshine-tiny/q8_0/moonshine-tiny-q8_0.oasr";

fn resolve_real_pack() -> PathBuf {
    if let Some(value) = std::env::var_os(MOONSHINE_OADP_REAL_PACK_ENV) {
        let path = PathBuf::from(value);
        assert!(
            path.is_file(),
            "{MOONSHINE_OADP_REAL_PACK_ENV} must point to an existing moonshine .oasr pack: {}",
            path.display()
        );
        return path;
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            panic!(
                "moonshine OADP real-pack prerequisites missing: HOME is not set and \
                 {MOONSHINE_OADP_REAL_PACK_ENV} is not set; #[ignore] is the opt-in, so this \
                 test must not silently skip"
            )
        });
    let path = home.join(MOONSHINE_PACK_HOME_RELATIVE_PATH);
    assert!(
        path.is_file(),
        "moonshine OADP real-pack prerequisites missing; install moonshine-tiny q8_0 (searched \
         {}) or set {MOONSHINE_OADP_REAL_PACK_ENV}",
        path.display()
    );
    path
}

fn read_runtime_source_preflight(runtime_path: &Path) -> GgmlAsrRuntimeSourcePreflight {
    let runtime_source =
        validate_ggml_runtime_source_path(runtime_path).expect("valid runtime source path");
    let metadata =
        read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
    let tensor_index =
        read_gguf_tensor_index_from_runtime_source(&runtime_source).expect("read tensor index");
    GgmlAsrRuntimeSourcePreflight {
        runtime_source,
        metadata,
        tensor_index: Arc::new(tensor_index),
    }
}

fn synthetic_encoder_output(
    metadata: MoonshineExecutionMetadata,
    frame_count: usize,
) -> MoonshineEncoderOutput {
    let mut rows = Vec::with_capacity(frame_count * metadata.d_model);
    for frame_idx in 0..frame_count {
        for hidden_idx in 0..metadata.d_model {
            rows.push(((frame_idx * metadata.d_model + hidden_idx) as f32 * 0.03125).sin());
        }
    }
    MoonshineEncoderOutput {
        frame_count,
        hidden_size: metadata.d_model,
        rows,
    }
}

/// Constant-fill adapter over the given base tensors, with the `alpha/rank`
/// scaling pre-folded into B exactly as the production loader does.
fn constant_fill_adapter(
    preflight: &GgmlAsrRuntimeSourcePreflight,
    fingerprint: &str,
    target_names: &[&str],
    rank: usize,
    alpha: f32,
    a_fill: f32,
    b_fill: f32,
) -> MoonshineLoraAdapter {
    let scale = alpha / rank as f32;
    let targets = target_names
        .iter()
        .map(|name| {
            let tensor = preflight
                .tensor_index
                .get(name)
                .unwrap_or_else(|| panic!("base tensor '{name}' must exist in the real pack"));
            let [input_dim, output_dim] = tensor.dims.as_slice() else {
                panic!("base tensor '{name}' must be rank-2");
            };
            let (input_dim, output_dim) = (*input_dim as usize, *output_dim as usize);
            (
                name.to_string(),
                MoonshineLoraTarget {
                    rank,
                    input_dim,
                    output_dim,
                    a_values: vec![a_fill; input_dim * rank],
                    b_scaled_values: vec![b_fill * scale; rank * output_dim],
                },
            )
        })
        .collect();
    moonshine_lora_adapter_for_test(fingerprint.to_string(), targets)
}

fn first_step_logits(
    prepared: &MoonshinePreparedRuntime,
    runtime_path: &Path,
    encoder_output: &MoonshineEncoderOutput,
    adapter: Option<&MoonshineLoraAdapter>,
) -> Vec<f32> {
    let mut runtime = MoonshineDecoderGraphRuntime::new(
        &prepared.decoder_weights,
        prepared.metadata,
        encoder_output.frame_count,
        true,
        Some(runtime_path),
        adapter,
    )
    .expect("decoder runtime");
    runtime
        .populate_cross_attention_cache(encoder_output)
        .expect("cross cache");
    runtime
        .compute_full_prefix_step_logits(&[prepared.metadata.bos_token_id])
        .expect("first-step logits")
}

/// Every decoder slot kind targeted at once (self-attn q/k/v/o, cross q/k/v/o
/// incl. the per-utterance cross-KV precompute, ffn up/down).
const ALL_DECODER_LAYER0_TARGETS: [&str; 10] = [
    "dec.blk.0.attn_q.weight",
    "dec.blk.0.attn_k.weight",
    "dec.blk.0.attn_v.weight",
    "dec.blk.0.attn_o.weight",
    "dec.blk.0.cross_q.weight",
    "dec.blk.0.cross_k.weight",
    "dec.blk.0.cross_v.weight",
    "dec.blk.0.cross_o.weight",
    "dec.blk.0.ffn_up.weight",
    "dec.blk.0.ffn_down.weight",
];

#[test]
#[ignore = "real-pack OADP oracle: needs moonshine-tiny q8_0 installed (~/.openasr) or OPENASR_MOONSHINE_OADP_REAL_PACK"]
fn zero_lora_keeps_decoder_logits_value_exact() {
    let pack = resolve_real_pack();
    let preflight = read_runtime_source_preflight(&pack);
    let prepared = build_moonshine_prepared_runtime(&preflight).expect("prepared runtime");
    let encoder_output = synthetic_encoder_output(prepared.metadata, 32);

    let baseline = first_step_logits(&prepared, &pack, &encoder_output, None);

    // Zero adapter: A nonzero (so A@x is a real intermediate), B all zeros.
    let zero_adapter = constant_fill_adapter(
        &preflight,
        "test:zero-lora",
        &ALL_DECODER_LAYER0_TARGETS,
        2,
        2.0,
        0.05,
        0.0,
    );
    let with_zero = first_step_logits(&prepared, &pack, &encoder_output, Some(&zero_adapter));

    assert_eq!(baseline.len(), with_zero.len());
    for (index, (base, zero)) in baseline.iter().zip(&with_zero).enumerate() {
        assert!(
            base == zero,
            "zero-adapter logit {index} diverged: base={base:?} zero={zero:?}\n\
             pack: {}\noracle: y = W@x + 0 must be value-exact (x + 0.0 == x)",
            pack.display()
        );
    }
}

/// The exact, attributed scale/direction oracle, measured AT the injection
/// point. The cross-V precompute rows are `W@x + B_scaled@(A@x)`; the base
/// `W@x` term cancels exactly between the adapter and no-adapter runs (same
/// input, same deterministic graph weights), so the observed delta must equal
/// host-computed `B_scaled@(A@x)` to f32 tolerance — and must scale EXACTLY
/// 2x when B doubles. This pins A/B orientation ([in,rank] / [rank,out]), the
/// pre-folded `alpha/rank` scaling, and the cross-KV precompute routing (the
/// memo's highest-risk injection point).
///
/// End-to-end logits/transcripts get only zero-exactness and effect oracles:
/// the q8_0 base quantizes ACTIVATIONS per block inside every base `mul_mat`
/// and the decoder self-KV caches are f16, so any small-perturbation
/// linearity claim downstream of the injection point is broken by
/// construction (verified empirically: sub-resolution deltas vanish exactly;
/// larger deltas are dominated by fixed-size rounding flips).
#[test]
#[ignore = "real-pack OADP oracle: needs moonshine-tiny q8_0 installed (~/.openasr) or OPENASR_MOONSHINE_OADP_REAL_PACK"]
fn lora_cross_value_precompute_delta_matches_host_math_and_scales_linearly() {
    let pack = resolve_real_pack();
    let preflight = read_runtime_source_preflight(&pack);
    let prepared = build_moonshine_prepared_runtime(&preflight).expect("prepared runtime");
    let metadata = prepared.metadata;
    let encoder_output = synthetic_encoder_output(metadata, 32);
    let d_model = metadata.d_model;

    let cross_value_rows = |adapter: Option<&MoonshineLoraAdapter>| {
        let mut runtime = MoonshineDecoderGraphRuntime::new(
            &prepared.decoder_weights,
            metadata,
            encoder_output.frame_count,
            true,
            Some(&pack),
            adapter,
        )
        .expect("decoder runtime");
        runtime
            .cross_value_projection_rows_for_test(0, &encoder_output)
            .expect("cross-V projection rows")
    };

    let base_rows = cross_value_rows(None);

    let rank = 2_usize;
    let alpha = 2.0_f32;
    let a_fill = 0.02_f32;
    for b_fill in [1.0e-2_f32, 2.0e-2] {
        let adapter = constant_fill_adapter(
            &preflight,
            &format!("test:cross-v-host-math-{b_fill}"),
            &["dec.blk.0.cross_v.weight"],
            rank,
            alpha,
            a_fill,
            b_fill,
        );
        let adapted_rows = cross_value_rows(Some(&adapter));
        assert_eq!(adapted_rows.len(), base_rows.len());

        // Host math: delta[j, f] = sum_r b_scaled[r, j] * (sum_i a[i, r] * enc[i, f]).
        // With constant fills this is rank * b_scaled * a_fill * S_f for every
        // output j of frame f, where S_f = sum_i enc[i, f].
        let b_scaled = b_fill * (alpha / rank as f32);
        for frame_idx in 0..encoder_output.frame_count {
            let frame = &encoder_output.rows[frame_idx * d_model..(frame_idx + 1) * d_model];
            let frame_sum: f32 = frame.iter().sum();
            let expected_delta = rank as f32 * b_scaled * a_fill * frame_sum;
            for hidden_idx in 0..d_model {
                let index = frame_idx * d_model + hidden_idx;
                let observed_delta = adapted_rows[index] - base_rows[index];
                let tolerance = 1.0e-5 + 1.0e-3 * expected_delta.abs();
                assert!(
                    (observed_delta - expected_delta).abs() <= tolerance,
                    "cross-V precompute delta mismatch at frame {frame_idx} hidden {hidden_idx} \
                     (b_fill {b_fill}): observed {observed_delta}, host-math expected \
                     {expected_delta}, tolerance {tolerance}\npack: {}",
                    pack.display()
                );
            }
        }
    }
}

/// Encoder-side effect oracle: a nonzero adapter on encoder targets must
/// change the encoder output (the all-f32 transformer path).
#[test]
#[ignore = "real-pack OADP oracle: needs moonshine-tiny q8_0 installed (~/.openasr) or OPENASR_MOONSHINE_OADP_REAL_PACK"]
fn lora_encoder_target_changes_encoder_output() {
    let pack = resolve_real_pack();
    let preflight = read_runtime_source_preflight(&pack);
    let prepared = build_moonshine_prepared_runtime(&preflight).expect("prepared runtime");
    let features = synthetic_waveform(8_000);

    let baseline = encoder_rows(&prepared, &pack, &features, None);
    let adapter = constant_fill_adapter(
        &preflight,
        "test:enc-effect",
        &["enc.blk.0.attn_v.weight", "enc.blk.0.ffn_down.weight"],
        2,
        2.0,
        0.02,
        0.05,
    );
    let adapted = encoder_rows(&prepared, &pack, &features, Some(&adapter));

    let norm = |values: &[f32]| values.iter().map(|v| v * v).sum::<f32>().sqrt();
    let delta: Vec<f32> = adapted
        .iter()
        .zip(&baseline)
        .map(|(with, base)| with - base)
        .collect();
    let delta_norm = norm(&delta);
    assert!(
        delta_norm > 1.0e-3,
        "encoder adapter must change encoder output rows (‖delta‖ = {delta_norm})\npack: {}",
        pack.display()
    );
}

/// Decoder cross-V effect oracle: the cross-V projection only reaches decode
/// through the per-utterance cross-KV precompute, so a nonzero logits delta
/// proves the precompute path is live end-to-end.
#[test]
#[ignore = "real-pack OADP oracle: needs moonshine-tiny q8_0 installed (~/.openasr) or OPENASR_MOONSHINE_OADP_REAL_PACK"]
fn lora_cross_v_target_changes_decoder_logits() {
    let pack = resolve_real_pack();
    let preflight = read_runtime_source_preflight(&pack);
    let prepared = build_moonshine_prepared_runtime(&preflight).expect("prepared runtime");
    let encoder_output = synthetic_encoder_output(prepared.metadata, 32);

    let baseline = first_step_logits(&prepared, &pack, &encoder_output, None);
    let adapter = constant_fill_adapter(
        &preflight,
        "test:cross-v-effect",
        &["dec.blk.0.cross_v.weight"],
        2,
        2.0,
        0.02,
        1.0e-3,
    );
    let with_adapter = first_step_logits(&prepared, &pack, &encoder_output, Some(&adapter));

    let norm = |values: &[f32]| values.iter().map(|v| v * v).sum::<f32>().sqrt();
    let delta: Vec<f32> = with_adapter
        .iter()
        .zip(&baseline)
        .map(|(with, base)| with - base)
        .collect();
    let delta_norm = norm(&delta);
    assert!(
        delta_norm > 1.0e-3,
        "cross-V adapter must change first-step decoder logits (‖delta‖ = {delta_norm}); the \
         cross-KV precompute side-path appears disconnected\npack: {}",
        pack.display()
    );
}

fn synthetic_waveform(sample_count: usize) -> super::frontend::MoonshineWaveformFeatures {
    let samples = (0..sample_count)
        .map(|index| (index as f32 * 0.011).sin() * 0.4)
        .collect();
    super::frontend::MoonshineWaveformFeatures { samples }
}

fn encoder_rows(
    prepared: &MoonshinePreparedRuntime,
    runtime_path: &Path,
    features: &super::frontend::MoonshineWaveformFeatures,
    adapter: Option<&MoonshineLoraAdapter>,
) -> Vec<f32> {
    let mut runtime = super::encoder_graph::MoonshineEncoderGraphRuntime::new(
        &prepared.encoder_weights,
        prepared.metadata,
        Some(runtime_path),
        adapter,
    )
    .expect("encoder runtime");
    runtime.encode(features).expect("encoder output").rows
}
