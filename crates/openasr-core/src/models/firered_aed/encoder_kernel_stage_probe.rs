//! Ignored real-pack harness: CPU vs Metal encoder taps, shipped kernel-stage
//! classifier, same-graph dual-output, and independent-runtime four-quadrant
//! A/B (C/D only when native ARGMAX_FIRST exists on that backend).

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{FireRedEncoderGraphRuntime, FireRedEncoderOutput, FireRedEncoderTapDump};
use crate::ggml_runtime::{
    DecodeFirstDivergenceClass, DiagnosticFamilyCompactPolicy,
    DiagnosticFourQuadrantClassificationInput, EncoderKernelStageChecksumPair,
    EncoderKernelStageClass, EncoderKernelStageClassification,
    EncoderKernelStageClassificationInput, EncoderKernelStageLayerChecksums,
    EncoderKernelStageStemChecksums, GgmlCpuGraphBackend, GgmlDecodeReuseMode,
    SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS, classify_encoder_kernel_stage,
    classify_four_quadrant_first_divergence, diagnostic_host_first_max_token,
    diagnostic_logits_sha256, diagnostic_top2, run_diagnostic_dual_output_conformance,
    run_diagnostic_four_quadrant_cpu_probe,
};
use crate::models::device_greedy_token::DeviceGreedyStepOutputMode;
use crate::models::firered_aed::decoder_graph::FireRedDecoderGraphRuntime;
use crate::models::firered_aed::frontend::{FireRedFbankFrontend, apply_cmvn};
use crate::models::firered_aed::runtime_contract::{
    FireRedAedExecutionMetadata, parse_firered_aed_execution_metadata,
};
use crate::models::firered_aed::tokenizer::FireRedTokenizer;
use crate::models::runtime_preflight::{
    build_runtime_tensor_reader_from_preflight, load_runtime_source_metadata_and_tensor_index,
};
use crate::models::seq2seq_decoder_state::{Seq2SeqDecoderState, Seq2SeqStateAxis};
use crate::models::seq2seq_greedy_decode::{
    Seq2SeqGreedyDecodeStepExecutor, Seq2SeqGreedyDecodeStepInput,
};

const TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const CMVN_NEG_MEAN_TENSOR: &str = "frontend.cmvn.neg_mean";
const CMVN_INV_STDDEV_TENSOR: &str = "frontend.cmvn.inv_stddev";

#[derive(Clone, Debug)]
struct StepTrace {
    step: u32,
    token_id: i32,
    logits_sha256: Option<String>,
    top2_margin: Option<f32>,
    graph_rebuilt: bool,
}

#[derive(Clone, Debug)]
struct QuadrantRun {
    lane: String,
    graph_mode: String,
    selection: String,
    reuse_requested: bool,
    reuse_actually_active: Option<bool>,
    tokens: Vec<i32>,
    steps: Vec<StepTrace>,
    transcript: Option<String>,
    error: Option<String>,
}

fn sha256_file(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_default();
    format!("{:x}", Sha256::digest(bytes))
}

fn class_name(class: DecodeFirstDivergenceClass) -> &'static str {
    match class {
        DecodeFirstDivergenceClass::ReusableKvOrOutputRefresh => "reusable_kv_or_output_refresh",
        DecodeFirstDivergenceClass::SelectorOrCompactOutput => "selector_or_compact_output",
        DecodeFirstDivergenceClass::EncoderCrossKvOrKernel => "encoder_cross_kv_or_kernel",
        DecodeFirstDivergenceClass::PersistentCompactInteraction => {
            "persistent_compact_interaction"
        }
        DecodeFirstDivergenceClass::EncoderCrossKvAllQuadrants => "encoder_cross_kv_all_quadrants",
        DecodeFirstDivergenceClass::NoneObserved => "none_observed",
        DecodeFirstDivergenceClass::InsufficientEvidence => "insufficient_evidence",
    }
}

fn kernel_stage_class_name(class: EncoderKernelStageClass) -> &'static str {
    match class {
        EncoderKernelStageClass::SubsampleInput => "subsample_input",
        EncoderKernelStageClass::SubsampleConvolution => "subsample_convolution",
        EncoderKernelStageClass::SubsampleBias => "subsample_bias",
        EncoderKernelStageClass::SubsampleRelu => "subsample_relu",
        EncoderKernelStageClass::SubsampleLayout => "subsample_layout",
        EncoderKernelStageClass::SubsampleOutputProjection => "subsample_output_projection",
        EncoderKernelStageClass::RelativePositionAttention => "relative_position_attention",
        EncoderKernelStageClass::DepthwiseConvolution => "depthwise_convolution",
        EncoderKernelStageClass::EncoderReadback => "encoder_readback",
        EncoderKernelStageClass::InsufficientEvidence => "insufficient_evidence",
    }
}

fn first_mismatch(left: &[i32], right: &[i32]) -> Option<usize> {
    let limit = left.len().min(right.len());
    (0..limit).find(|&index| left[index] != right[index]).or({
        if left.len() == right.len() {
            None
        } else {
            Some(limit)
        }
    })
}

fn decoder_state(
    metadata: FireRedAedExecutionMetadata,
    cross_positions: usize,
) -> Seq2SeqDecoderState {
    let budget = super::super::decode_budget::firered_aed_decode_budget(
        cross_positions,
        metadata.decoder_pe_len,
    )
    .expect("firered-aed decode budget");
    Seq2SeqDecoderState {
        self_attention: Seq2SeqStateAxis {
            logical_positions: budget.self_kv_positions,
            resident_positions: budget.self_kv_positions,
            hard_position_cap: metadata.decoder_pe_len,
        },
        cross_attention: Seq2SeqStateAxis {
            logical_positions: cross_positions,
            resident_positions: cross_positions,
            hard_position_cap: metadata.encoder_max_frames(),
        },
    }
}

fn new_decoder(
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    metadata: FireRedAedExecutionMetadata,
    encoder_frames: usize,
    backend: GgmlCpuGraphBackend,
    output_mode: DeviceGreedyStepOutputMode,
    reuse_mode: GgmlDecodeReuseMode,
) -> Result<FireRedDecoderGraphRuntime, String> {
    FireRedDecoderGraphRuntime::new_with_greedy_step_output_mode(
        preflight,
        metadata,
        decoder_state(metadata, encoder_frames),
        backend,
        output_mode,
        reuse_mode,
    )
    .map_err(|error| error.to_string())
}

fn skipped(lane: &str, graph_mode: &str, selection: &str, reason: &str) -> QuadrantRun {
    QuadrantRun {
        lane: lane.to_string(),
        graph_mode: graph_mode.to_string(),
        selection: selection.to_string(),
        reuse_requested: graph_mode == "reusable_graph",
        reuse_actually_active: None,
        tokens: Vec::new(),
        steps: Vec::new(),
        transcript: None,
        error: Some(reason.to_string()),
    }
}

fn decode_tokens(
    tokens: &[i32],
    metadata: FireRedAedExecutionMetadata,
    tokenizer: &FireRedTokenizer,
) -> Option<String> {
    tokenizer
        .decode(
            &tokens
                .iter()
                .copied()
                .filter(|token| *token as u32 != metadata.eos_token_id)
                .map(|token| token as u32)
                .collect::<Vec<_>>(),
        )
        .ok()
        .map(|text| text.trim().to_string())
}

fn decode_full_logits(
    runtime: &mut FireRedDecoderGraphRuntime,
    metadata: FireRedAedExecutionMetadata,
    encoder: &FireRedEncoderOutput,
    tokenizer: &FireRedTokenizer,
    force_fresh: bool,
    max_steps: usize,
) -> Result<(QuadrantRun, Vec<Vec<f32>>), String> {
    runtime
        .populate_cross_attention_cache(&encoder.rows, encoder.frame_count)
        .map_err(|error| error.to_string())?;
    let mut prefix = vec![metadata.sos_token_id];
    let mut tokens = Vec::new();
    let mut steps = Vec::new();
    let mut logits_rows = Vec::new();
    for step in 0..max_steps {
        let logits = if force_fresh {
            runtime
                .compute_step_logits_forcing_fresh_graph(&prefix)
                .map_err(|error| format!("fresh logits step {step}: {error}"))?
        } else {
            runtime
                .compute_step_logits(&prefix)
                .map_err(|error| format!("logits step {step}: {error}"))?
        };
        let token = diagnostic_host_first_max_token(&logits)
            .ok_or_else(|| format!("step {step}: host first-max found no finite logit"))?;
        steps.push(StepTrace {
            step: u32::try_from(step).unwrap_or(u32::MAX),
            token_id: token,
            logits_sha256: Some(diagnostic_logits_sha256(&logits)),
            top2_margin: diagnostic_top2(&logits).and_then(|top2| top2.margin),
            graph_rebuilt: force_fresh || !runtime.has_active_reuse_graph(),
        });
        tokens.push(token);
        logits_rows.push(logits);
        if token as u32 == metadata.eos_token_id {
            break;
        }
        prefix.push(token as u32);
    }
    Ok((
        QuadrantRun {
            lane: String::new(),
            graph_mode: if force_fresh {
                "fresh_rebuild".to_string()
            } else {
                "reusable_graph".to_string()
            },
            selection: "complete_logits_host_first_max".to_string(),
            reuse_requested: !force_fresh,
            reuse_actually_active: Some(runtime.has_active_reuse_graph()),
            transcript: decode_tokens(&tokens, metadata, tokenizer),
            tokens,
            steps,
            error: None,
        },
        logits_rows,
    ))
}

fn decode_native_compact(
    runtime: &mut FireRedDecoderGraphRuntime,
    metadata: FireRedAedExecutionMetadata,
    encoder: &FireRedEncoderOutput,
    tokenizer: &FireRedTokenizer,
    max_steps: usize,
) -> Result<QuadrantRun, String> {
    runtime
        .populate_cross_attention_cache(&encoder.rows, encoder.frame_count)
        .map_err(|error| error.to_string())?;
    let mut generated = Vec::new();
    let mut tokens = Vec::new();
    let mut steps = Vec::new();
    for step in 0..max_steps {
        let output = runtime
            .decode_step_logits(Seq2SeqGreedyDecodeStepInput {
                initial_prompt_tokens: std::slice::from_ref(&metadata.sos_token_id),
                generated_tokens: &generated,
                step_index: step,
            })
            .map_err(|error| format!("compact step {step}: {error}"))?;
        let token = if let Some(hint) = output.greedy_token_hint {
            i32::try_from(hint).map_err(|_| "compact token does not fit i32".to_string())?
        } else if !output.logits.is_empty() {
            diagnostic_host_first_max_token(&output.logits)
                .ok_or_else(|| format!("compact step {step}: no finite logit"))?
        } else {
            return Err(format!("compact step {step}: no hint and no logits"));
        };
        steps.push(StepTrace {
            step: u32::try_from(step).unwrap_or(u32::MAX),
            token_id: token,
            logits_sha256: if output.logits.is_empty() {
                None
            } else {
                Some(diagnostic_logits_sha256(&output.logits))
            },
            top2_margin: if output.logits.is_empty() {
                None
            } else {
                diagnostic_top2(&output.logits).and_then(|top2| top2.margin)
            },
            graph_rebuilt: !runtime.has_active_reuse_graph(),
        });
        tokens.push(token);
        if token as u32 == metadata.eos_token_id {
            break;
        }
        generated.push(token as u32);
    }
    Ok(QuadrantRun {
        lane: "cpu".to_string(),
        graph_mode: "fresh_rebuild".to_string(),
        selection: "native_argmax_first".to_string(),
        reuse_requested: false,
        reuse_actually_active: Some(runtime.has_active_reuse_graph()),
        transcript: decode_tokens(&tokens, metadata, tokenizer),
        tokens,
        steps,
        error: None,
    })
}

fn run_named(
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    metadata: FireRedAedExecutionMetadata,
    tokenizer: &FireRedTokenizer,
    lane: &str,
    backend: GgmlCpuGraphBackend,
    encoder: &FireRedEncoderOutput,
    output_mode: DeviceGreedyStepOutputMode,
    reuse_mode: GgmlDecodeReuseMode,
    force_fresh: bool,
    compact: bool,
    max_steps: usize,
) -> QuadrantRun {
    eprintln!(
        "decode lane={lane} backend={backend:?} mode={output_mode:?} reuse={reuse_mode:?} fresh={force_fresh} compact={compact}"
    );
    let graph_mode = if force_fresh {
        "fresh_rebuild"
    } else {
        "reusable_graph"
    };
    let selection = if compact {
        "native_argmax_first"
    } else {
        "complete_logits_host_first_max"
    };
    let mut runtime = match new_decoder(
        preflight,
        metadata,
        encoder.frame_count,
        backend,
        output_mode,
        reuse_mode,
    ) {
        Ok(runtime) => runtime,
        Err(error) => return skipped(lane, graph_mode, selection, &error),
    };
    let mut run = if compact {
        match decode_native_compact(&mut runtime, metadata, encoder, tokenizer, max_steps) {
            Ok(run) => run,
            Err(error) => skipped(lane, graph_mode, selection, &error),
        }
    } else {
        match decode_full_logits(
            &mut runtime,
            metadata,
            encoder,
            tokenizer,
            force_fresh,
            max_steps,
        ) {
            Ok((run, _)) => run,
            Err(error) => skipped(lane, graph_mode, selection, &error),
        }
    };
    run.lane = lane.to_string();
    run
}

fn quadrant_json(run: &QuadrantRun) -> Value {
    json!({
        "lane": run.lane,
        "graph_mode": run.graph_mode,
        "selection": run.selection,
        "reuse_requested": run.reuse_requested,
        "reuse_actually_active": run.reuse_actually_active,
        "tokens": run.tokens,
        "transcript": run.transcript,
        "error": run.error,
        "steps": run.steps.iter().map(|step| json!({
            "step": step.step,
            "token_id": step.token_id,
            "logits_sha256": step.logits_sha256,
            "top2_margin": step.top2_margin,
            "graph_rebuilt": step.graph_rebuilt,
        })).collect::<Vec<_>>(),
    })
}

fn encode_on(
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    metadata: FireRedAedExecutionMetadata,
    backend: GgmlCpuGraphBackend,
    features: &[f32],
    n_frames: usize,
) -> Result<FireRedEncoderOutput, String> {
    let mut encoder = FireRedEncoderGraphRuntime::new(preflight, metadata, backend)
        .map_err(|error| error.to_string())?;
    encoder
        .encode(features, n_frames)
        .map_err(|error| error.to_string())
}

fn taps_on(
    preflight: &crate::ggml_runtime::GgufRuntimeSourcePreflight,
    metadata: FireRedAedExecutionMetadata,
    backend: GgmlCpuGraphBackend,
    features: &[f32],
    n_frames: usize,
    tap_layer_idx: Option<usize>,
) -> Result<FireRedEncoderTapDump, String> {
    let mut encoder = FireRedEncoderGraphRuntime::new(preflight, metadata, backend)
        .map_err(|error| error.to_string())?;
    encoder
        .encode_with_layer_taps(features, n_frames, tap_layer_idx)
        .map_err(|error| error.to_string())
}

fn default_pack_path() -> PathBuf {
    if let Ok(path) = std::env::var("OPENASR_FIRERED_PROBE_PACK") {
        return PathBuf::from(path);
    }
    if let Ok(home) = std::env::var("OPENASR_HOME") {
        let pack = PathBuf::from(home).join("models/packs/firered-aed-l-v2-q4_k.oasr");
        if pack.exists() {
            return pack;
        }
    }
    PathBuf::from(
        "/var/folders/4m/2gh64f9n09g1mlx3qvtmqzpw0000gn/T/grok-goal-90413312db24/scratch/openasr-home/models/packs/firered-aed-l-v2-q4_k.oasr",
    )
}

fn default_audio_path() -> PathBuf {
    if let Ok(path) = std::env::var("OPENASR_FIRERED_PROBE_AUDIO") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
}

fn default_out_path() -> PathBuf {
    if let Ok(path) = std::env::var("OPENASR_FIRERED_PROBE_OUT") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/design/gpu-decode-correctness-evidence/firered-four-quadrant.json")
}

fn core_commit_label() -> String {
    std::env::var("OPENASR_FIRERED_PROBE_COMMIT")
        .unwrap_or_else(|_| "PENDING_CLASSIFIER_COMMIT".to_string())
}

struct OwnedTapPair {
    cpu: String,
    accel: String,
}

impl OwnedTapPair {
    fn pair(&self) -> EncoderKernelStageChecksumPair<'_> {
        EncoderKernelStageChecksumPair {
            cpu: &self.cpu,
            accel: &self.accel,
        }
    }
}

struct OwnedStemTapPairs {
    mel_4d: OwnedTapPair,
    conv1_raw: OwnedTapPair,
    conv1_bias: OwnedTapPair,
    conv1_relu: OwnedTapPair,
    conv2_raw: OwnedTapPair,
    conv2_bias: OwnedTapPair,
    conv2_relu: OwnedTapPair,
    after_permute: OwnedTapPair,
    after_cont: OwnedTapPair,
    flat_2d: OwnedTapPair,
    out_matmul: OwnedTapPair,
    subsample_out: OwnedTapPair,
}

impl OwnedStemTapPairs {
    fn from_dumps(cpu: &FireRedEncoderTapDump, accel: &FireRedEncoderTapDump) -> Self {
        let pair = |cpu: &[f32], accel: &[f32]| OwnedTapPair {
            cpu: diagnostic_logits_sha256(cpu),
            accel: diagnostic_logits_sha256(accel),
        };
        Self {
            mel_4d: pair(&cpu.stem.mel_4d, &accel.stem.mel_4d),
            conv1_raw: pair(&cpu.stem.conv1_raw, &accel.stem.conv1_raw),
            conv1_bias: pair(&cpu.stem.conv1_bias, &accel.stem.conv1_bias),
            conv1_relu: pair(&cpu.stem.conv1_relu, &accel.stem.conv1_relu),
            conv2_raw: pair(&cpu.stem.conv2_raw, &accel.stem.conv2_raw),
            conv2_bias: pair(&cpu.stem.conv2_bias, &accel.stem.conv2_bias),
            conv2_relu: pair(&cpu.stem.conv2_relu, &accel.stem.conv2_relu),
            after_permute: pair(&cpu.stem.after_permute, &accel.stem.after_permute),
            after_cont: pair(&cpu.stem.after_cont, &accel.stem.after_cont),
            flat_2d: pair(&cpu.stem.flat_2d, &accel.stem.flat_2d),
            out_matmul: pair(&cpu.stem.out_matmul, &accel.stem.out_matmul),
            subsample_out: pair(&cpu.stem.subsample_out, &accel.stem.subsample_out),
        }
    }

    fn checksums(&self) -> EncoderKernelStageStemChecksums<'_> {
        EncoderKernelStageStemChecksums {
            mel_4d: Some(self.mel_4d.pair()),
            conv1_raw: Some(self.conv1_raw.pair()),
            conv1_bias: Some(self.conv1_bias.pair()),
            conv1_relu: Some(self.conv1_relu.pair()),
            conv2_raw: Some(self.conv2_raw.pair()),
            conv2_bias: Some(self.conv2_bias.pair()),
            conv2_relu: Some(self.conv2_relu.pair()),
            after_permute: Some(self.after_permute.pair()),
            after_cont: Some(self.after_cont.pair()),
            flat_2d: Some(self.flat_2d.pair()),
            out_matmul: Some(self.out_matmul.pair()),
            subsample_out: Some(self.subsample_out.pair()),
        }
    }

    fn json(&self) -> Value {
        let pair = |pair: &OwnedTapPair| {
            json!({
                "cpu": &pair.cpu,
                "metal": &pair.accel,
                "matches": pair.cpu == pair.accel,
            })
        };
        json!({
            "mel_4d": pair(&self.mel_4d),
            "conv1_raw": pair(&self.conv1_raw),
            "conv1_bias": pair(&self.conv1_bias),
            "conv1_relu": pair(&self.conv1_relu),
            "conv2_raw": pair(&self.conv2_raw),
            "conv2_bias": pair(&self.conv2_bias),
            "conv2_relu": pair(&self.conv2_relu),
            "after_permute": pair(&self.after_permute),
            "after_cont": pair(&self.after_cont),
            "flat_2d": pair(&self.flat_2d),
            "out_matmul": pair(&self.out_matmul),
            "subsample_out": pair(&self.subsample_out),
        })
    }
}

fn classify_owned(
    stem: &OwnedStemTapPairs,
    layers: &[(usize, OwnedTapPair, Option<[OwnedTapPair; 4]>)],
    encoder_output: &OwnedTapPair,
) -> EncoderKernelStageClassification {
    let layer_views: Vec<EncoderKernelStageLayerChecksums<'_>> = layers
        .iter()
        .map(|(index, block, intra)| {
            let (ffn1, attn, conv, ffn2) = match intra.as_ref() {
                Some([ffn1, attn, conv, ffn2]) => (
                    Some(ffn1.pair()),
                    Some(attn.pair()),
                    Some(conv.pair()),
                    Some(ffn2.pair()),
                ),
                None => (None, None, None, None),
            };
            EncoderKernelStageLayerChecksums {
                layer_index: *index,
                ffn1_out: ffn1,
                attn_out: attn,
                conv_out: conv,
                ffn2_out: ffn2,
                block_out: Some(block.pair()),
            }
        })
        .collect();
    classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
        stem: Some(stem.checksums()),
        subsample: Some(stem.subsample_out.pair()),
        layers: &layer_views,
        encoder_output: Some(encoder_output.pair()),
    })
}

#[test]
#[ignore = "requires isolated firered-aed-l-v2-q4_k.oasr pack; writes four-quadrant evidence JSON"]
fn firered_cpu_metal_encoder_kernel_stage_and_four_quadrant() {
    let started = Instant::now();
    let pack_path = default_pack_path();
    let audio_path = default_audio_path();
    let out_path = default_out_path();
    let max_steps = SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS;

    if !pack_path.exists() {
        eprintln!("skipping: pack missing at {}", pack_path.display());
        return;
    }
    if !audio_path.exists() {
        eprintln!("skipping: audio missing at {}", audio_path.display());
        return;
    }

    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        &audio_path,
        "firered four-quadrant probe",
        "firered four-quadrant probe",
    )
    .expect("load jfk.wav");
    let preflight =
        load_runtime_source_metadata_and_tensor_index(&pack_path).expect("read gguf preflight");
    let metadata = parse_firered_aed_execution_metadata(preflight.metadata.as_ref())
        .expect("parse firered metadata");
    let tokenizer = FireRedTokenizer::new(
        preflight
            .metadata
            .get_string_array(TOKENIZER_TOKENS_KEY)
            .expect("pack missing tokenizer.ggml.tokens")
            .to_vec(),
    );
    let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
    let feature_dim = [metadata.feature_dim as u64];
    let neg_mean = reader
        .host_tensor_f32_copy_by_name(CMVN_NEG_MEAN_TENSOR, &feature_dim)
        .expect("cmvn neg_mean");
    let inv_stddev = reader
        .host_tensor_f32_copy_by_name(CMVN_INV_STDDEV_TENSOR, &feature_dim)
        .expect("cmvn inv_stddev");
    let frontend = FireRedFbankFrontend::new();
    let mut features = frontend.compute(&samples).expect("fbank");
    apply_cmvn(&mut features.data, features.n_mels, &neg_mean, &inv_stddev).expect("cmvn");

    eprintln!("independent encode CPU...");
    let cpu_encoder = encode_on(
        &preflight,
        metadata,
        GgmlCpuGraphBackend::Cpu,
        &features.data,
        features.n_frames,
    )
    .expect("cpu encode");
    eprintln!("independent encode Metal...");
    let metal_encoder = encode_on(
        &preflight,
        metadata,
        GgmlCpuGraphBackend::Metal,
        &features.data,
        features.n_frames,
    )
    .expect("metal encode");

    eprintln!("layer taps CPU (all block_out)...");
    let cpu_blocks = taps_on(
        &preflight,
        metadata,
        GgmlCpuGraphBackend::Cpu,
        &features.data,
        features.n_frames,
        None,
    )
    .expect("cpu block taps");
    eprintln!("layer taps Metal (all block_out)...");
    let metal_blocks = taps_on(
        &preflight,
        metadata,
        GgmlCpuGraphBackend::Metal,
        &features.data,
        features.n_frames,
        None,
    )
    .expect("metal block taps");

    let stem = OwnedStemTapPairs::from_dumps(&cpu_blocks, &metal_blocks);
    let encoder_output = OwnedTapPair {
        cpu: diagnostic_logits_sha256(&cpu_encoder.rows),
        accel: diagnostic_logits_sha256(&metal_encoder.rows),
    };

    let mut first_block_layer = None;
    let layer_count = cpu_blocks
        .block_rows
        .len()
        .min(metal_blocks.block_rows.len());
    let mut layer_block_pairs = Vec::with_capacity(layer_count);
    for index in 0..layer_count {
        let pair = OwnedTapPair {
            cpu: diagnostic_logits_sha256(&cpu_blocks.block_rows[index]),
            accel: diagnostic_logits_sha256(&metal_blocks.block_rows[index]),
        };
        if first_block_layer.is_none() && pair.cpu != pair.accel {
            first_block_layer = Some(index);
        }
        layer_block_pairs.push(pair);
    }

    let mut intra_pairs: Option<[OwnedTapPair; 4]> = None;
    if let Some(layer) = first_block_layer {
        eprintln!("intra taps CPU layer {layer}...");
        let cpu_intra = taps_on(
            &preflight,
            metadata,
            GgmlCpuGraphBackend::Cpu,
            &features.data,
            features.n_frames,
            Some(layer),
        )
        .expect("cpu intra taps");
        eprintln!("intra taps Metal layer {layer}...");
        let metal_intra = taps_on(
            &preflight,
            metadata,
            GgmlCpuGraphBackend::Metal,
            &features.data,
            features.n_frames,
            Some(layer),
        )
        .expect("metal intra taps");
        let cpu_taps = cpu_intra.intra_taps.as_ref().expect("cpu intra present");
        let metal_taps = metal_intra
            .intra_taps
            .as_ref()
            .expect("metal intra present");
        intra_pairs = Some([
            OwnedTapPair {
                cpu: diagnostic_logits_sha256(&cpu_taps.ffn1_out),
                accel: diagnostic_logits_sha256(&metal_taps.ffn1_out),
            },
            OwnedTapPair {
                cpu: diagnostic_logits_sha256(&cpu_taps.attn_out),
                accel: diagnostic_logits_sha256(&metal_taps.attn_out),
            },
            OwnedTapPair {
                cpu: diagnostic_logits_sha256(&cpu_taps.conv_out),
                accel: diagnostic_logits_sha256(&metal_taps.conv_out),
            },
            OwnedTapPair {
                cpu: diagnostic_logits_sha256(&cpu_taps.ffn2_out),
                accel: diagnostic_logits_sha256(&metal_taps.ffn2_out),
            },
        ]);
    }

    let mut layers_owned: Vec<(usize, OwnedTapPair, Option<[OwnedTapPair; 4]>)> = Vec::new();
    for (index, block) in layer_block_pairs.into_iter().enumerate() {
        let intra = if Some(index) == first_block_layer {
            intra_pairs.take()
        } else {
            None
        };
        layers_owned.push((index, block, intra));
    }
    let classification = classify_owned(&stem, &layers_owned, &encoder_output);

    eprintln!(
        "encoder_kernel_stage class={} layer={:?} tap={:?}",
        kernel_stage_class_name(classification.class),
        classification.first_divergent_layer,
        classification.first_divergent_tap
    );

    eprintln!("CPU A fresh FullLogits...");
    let mut cpu_a_runtime = new_decoder(
        &preflight,
        metadata,
        cpu_encoder.frame_count,
        GgmlCpuGraphBackend::Cpu,
        DeviceGreedyStepOutputMode::FullLogits,
        GgmlDecodeReuseMode::FreshGraph,
    )
    .expect("cpu A runtime");
    let (mut cpu_a, cpu_a_logits) = decode_full_logits(
        &mut cpu_a_runtime,
        metadata,
        &cpu_encoder,
        &tokenizer,
        true,
        max_steps,
    )
    .expect("cpu A decode");
    cpu_a.lane = "cpu".to_string();
    drop(cpu_a_runtime);

    let cpu_b = skipped(
        "cpu",
        "reusable_graph",
        "complete_logits_host_first_max",
        "CPU direct execution does not support reusable in-place KV; B would silently rebuild and is not reusable-KV evidence",
    );
    let cpu_c = run_named(
        &preflight,
        metadata,
        &tokenizer,
        "cpu",
        GgmlCpuGraphBackend::Cpu,
        &cpu_encoder,
        DeviceGreedyStepOutputMode::DeviceTop1,
        GgmlDecodeReuseMode::FreshGraph,
        true,
        true,
        max_steps,
    );
    let cpu_d = skipped(
        "cpu",
        "reusable_graph",
        "native_argmax_first",
        "CPU reusable-KV is not a proven lane; C/D compact reuse omitted",
    );
    let metal_a = run_named(
        &preflight,
        metadata,
        &tokenizer,
        "metal",
        GgmlCpuGraphBackend::Metal,
        &metal_encoder,
        DeviceGreedyStepOutputMode::FullLogits,
        GgmlDecodeReuseMode::FreshGraph,
        true,
        false,
        max_steps,
    );
    let metal_b = run_named(
        &preflight,
        metadata,
        &tokenizer,
        "metal",
        GgmlCpuGraphBackend::Metal,
        &metal_encoder,
        DeviceGreedyStepOutputMode::FullLogits,
        GgmlDecodeReuseMode::ReusableGraph,
        false,
        false,
        max_steps,
    );
    let metal_c = skipped(
        "metal",
        "fresh_rebuild",
        "native_argmax_first",
        "Metal has no native ARGMAX_FIRST; C omitted, not a pass",
    );
    let metal_d = skipped(
        "metal",
        "reusable_graph",
        "native_argmax_first",
        "Metal has no native ARGMAX_FIRST; D omitted, not a pass",
    );

    let logit_refs: Vec<&[f32]> = cpu_a_logits.iter().map(|row| row.as_slice()).collect();
    let dual_output = match run_diagnostic_dual_output_conformance(&logit_refs) {
        Ok(results) => json!({
            "authorizes_production_compact": false,
            "ran": true,
            "graph": "diagnostic_same_graph_logits_plus_argmax_first",
            "source_logits": "cpu_fresh_full_logits_firered_decoder",
            "all_tokens_match": results.iter().all(|result| result.tokens_match),
            "steps": results.iter().enumerate().map(|(step, result)| json!({
                "step": step,
                "device_token": result.device_token,
                "host_first_max_token": result.host_first_max_token,
                "tokens_match": result.tokens_match,
                "authorizes_production_compact": result.authorizes_production_compact(),
                "top2_margin": result.top2.margin,
                "logits_sha256": diagnostic_logits_sha256(&result.logits),
            })).collect::<Vec<_>>(),
        }),
        Err(error) => json!({
            "authorizes_production_compact": false,
            "ran": false,
            "error": error.to_string(),
        }),
    };

    let synthetic_selector = if logit_refs.is_empty() {
        json!({ "ran": false })
    } else {
        match run_diagnostic_four_quadrant_cpu_probe(
            &logit_refs,
            DiagnosticFamilyCompactPolicy::NativeArgmaxFirstEligible,
        ) {
            Ok(synthetic) => json!({
                "ran": true,
                "note": "synthetic ggml rows from captured FireRed CPU-A logits; not a second FireRed runtime",
                "classification": class_name(synthetic.classification),
            }),
            Err(error) => json!({ "ran": false, "error": error.to_string() }),
        }
    };

    let metal_class = if metal_a.error.is_none() && metal_b.error.is_none() {
        Some(classify_four_quadrant_first_divergence(
            DiagnosticFourQuadrantClassificationInput {
                case_a: &metal_a.tokens,
                case_b: &metal_b.tokens,
                case_c: None,
                case_d: None,
                cpu_reference: Some(&cpu_a.tokens),
            },
        ))
    } else {
        None
    };
    let cpu_class = if cpu_c.error.is_none() {
        Some(classify_four_quadrant_first_divergence(
            DiagnosticFourQuadrantClassificationInput {
                case_a: &cpu_a.tokens,
                case_b: &cpu_a.tokens,
                case_c: Some(&cpu_c.tokens),
                case_d: None,
                cpu_reference: Some(&cpu_a.tokens),
            },
        ))
    } else {
        None
    };

    let encoder_match = encoder_output.cpu == encoder_output.accel;
    let mut first_class = DecodeFirstDivergenceClass::InsufficientEvidence;
    let mut rationale = Vec::new();
    if !encoder_match {
        first_class = DecodeFirstDivergenceClass::EncoderCrossKvOrKernel;
        rationale.push(format!(
            "CPU vs Metal encoder checksums differ; shipped classifier={} layer={:?} tap={:?}",
            kernel_stage_class_name(classification.class),
            classification.first_divergent_layer,
            classification.first_divergent_tap
        ));
    } else if first_mismatch(&cpu_a.tokens, &metal_a.tokens) == Some(0) {
        first_class = DecodeFirstDivergenceClass::EncoderCrossKvOrKernel;
        rationale.push("Metal fresh FullLogits diverges from CPU A at step 0".to_string());
    } else if metal_a.error.is_none()
        && metal_b.error.is_none()
        && cpu_a.tokens == metal_a.tokens
        && metal_a.tokens != metal_b.tokens
    {
        first_class = DecodeFirstDivergenceClass::ReusableKvOrOutputRefresh;
        rationale.push("Metal A matches CPU A; Metal reusable B diverges".to_string());
    } else if cpu_c.error.is_none() && cpu_a.tokens != cpu_c.tokens {
        first_class = DecodeFirstDivergenceClass::SelectorOrCompactOutput;
        rationale.push("CPU native compact C diverges from CPU FullLogits A".to_string());
    } else if metal_a.error.is_none()
        && metal_b.error.is_none()
        && cpu_c.error.is_none()
        && cpu_a.tokens == metal_a.tokens
        && metal_a.tokens == metal_b.tokens
        && cpu_a.tokens == cpu_c.tokens
    {
        first_class = DecodeFirstDivergenceClass::NoneObserved;
        rationale.push("CPU A/C and Metal A/B token sequences agree".to_string());
    } else if let Some(class) = metal_class {
        first_class = class;
        rationale.push(format!(
            "fallback to metal-lane helper classification {}",
            class_name(class)
        ));
    }
    rationale
        .push("CUDA/Vulkan/HIP were not activated; compact GPU C/D remain untested".to_string());
    rationale.push("dual-output success does not authorize production compact".to_string());

    let intra_json = first_block_layer.and_then(|layer| {
        layers_owned.get(layer).and_then(|(_, block, intra)| {
            intra.as_ref().map(|taps| {
                json!({
                    "layer": layer,
                    "ffn1_out": { "cpu": taps[0].cpu, "metal": taps[0].accel, "matches": taps[0].cpu == taps[0].accel },
                    "attn_out": { "cpu": taps[1].cpu, "metal": taps[1].accel, "matches": taps[1].cpu == taps[1].accel },
                    "conv_out": { "cpu": taps[2].cpu, "metal": taps[2].accel, "matches": taps[2].cpu == taps[2].accel },
                    "ffn2_out": { "cpu": taps[3].cpu, "metal": taps[3].accel, "matches": taps[3].cpu == taps[3].accel },
                    "block_out": { "cpu": block.cpu, "metal": block.accel, "matches": block.cpu == block.accel },
                })
            })
        })
    });

    let report = json!({
        "schema": "openasr.firered-four-quadrant.v0",
        "core_commit": core_commit_label(),
        "authorizes_production_compact": false,
        "notes": [
            "dual-output is a diagnostic graph on captured FireRed logits; agreement does not authorize production compact",
            "Metal stays FullLogits; CPU C uses native ARGMAX_FIRST; CPU D omitted because reusable-KV is unsupported on CPU",
            "Metal C/D omitted because Metal has no native ARGMAX_FIRST; omitted is not a pass",
            "encoder_kernel_stage is the result of shipped classify_encoder_kernel_stage, not a hand fill",
            "CUDA/Vulkan/HIP were not tested on this host",
        ],
        "pack": {
            "path_label": "isolated-openasr-home/models/packs/firered-aed-l-v2-q4_k.oasr",
            "model_id": "firered-aed-l-v2",
            "quant": "q4_k",
            "sha256": sha256_file(&pack_path),
            "size_bytes": std::fs::metadata(&pack_path).map(|meta| meta.len()).unwrap_or(0),
        },
        "audio": {
            "path_or_label": "fixtures/jfk.wav",
            "sha256": sha256_file(&audio_path),
            "duration_s": 11.0,
        },
        "encoder": {
            "cpu": {
                "frame_count": cpu_encoder.frame_count,
                "hidden_size": cpu_encoder.hidden_size,
                "checksum": encoder_output.cpu,
            },
            "metal": {
                "frame_count": metal_encoder.frame_count,
                "hidden_size": metal_encoder.hidden_size,
                "checksum": encoder_output.accel,
                "matches_cpu": encoder_match,
            },
            "subsample": {
                "cpu": &stem.subsample_out.cpu,
                "metal": &stem.subsample_out.accel,
                "matches": stem.subsample_out.cpu == stem.subsample_out.accel,
            },
            "stem_taps": stem.json(),
        },
        "encoder_layer_taps": {
            "first_divergent_block_layer": first_block_layer,
            "intra": intra_json,
            "block_out": layers_owned.iter().map(|(index, block, _)| json!({
                "layer": index,
                "cpu": block.cpu,
                "metal": block.accel,
                "matches": block.cpu == block.accel,
            })).collect::<Vec<_>>(),
        },
        "encoder_kernel_stage": {
            "class": kernel_stage_class_name(classification.class),
            "first_divergent_layer": classification.first_divergent_layer,
            "first_divergent_tap": classification.first_divergent_tap,
            "source": "classify_encoder_kernel_stage",
        },
        "dual_output": dual_output,
        "synthetic_selector_four_quadrant": synthetic_selector,
        "quadrants": {
            "cpu_a": quadrant_json(&cpu_a),
            "cpu_b": quadrant_json(&cpu_b),
            "cpu_c": quadrant_json(&cpu_c),
            "cpu_d": quadrant_json(&cpu_d),
            "metal_a": quadrant_json(&metal_a),
            "metal_b": quadrant_json(&metal_b),
            "metal_c": quadrant_json(&metal_c),
            "metal_d": quadrant_json(&metal_d),
        },
        "first_divergence": {
            "class": class_name(first_class),
            "metal_lane_helper": metal_class.map(class_name),
            "cpu_lane_helper": cpu_class.map(class_name),
            "cpu_vs_metal_a_first_mismatch": first_mismatch(&cpu_a.tokens, &metal_a.tokens),
            "metal_a_vs_b_first_mismatch": first_mismatch(&metal_a.tokens, &metal_b.tokens),
            "cpu_a_vs_c_first_mismatch": first_mismatch(&cpu_a.tokens, &cpu_c.tokens),
            "cpu_encoder_vs_metal_encoder_match": encoder_match,
            "dual_output_tokens_match": dual_output.get("all_tokens_match").and_then(Value::as_bool),
            "dual_output_authorizes_production_compact": false,
            "rationale": rationale,
        },
        "untested": {
            "cuda": "not activated; no compact GPU C/D",
            "vulkan": "not activated",
            "hip": "not activated",
        },
        "elapsed_s": started.elapsed().as_secs_f64(),
    });

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("create evidence dir");
    }
    std::fs::write(
        &out_path,
        serde_json::to_vec_pretty(&report).expect("serialize evidence"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", out_path.display()));
    eprintln!("wrote {}", out_path.display());
}
