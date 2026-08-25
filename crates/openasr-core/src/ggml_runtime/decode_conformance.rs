//! Diagnostic / conformance probes for the GPU decode correctness contract.
//!
//! These helpers gather shared exact-route Layer-1/Layer-2 diagnostic evidence,
//! including inside the isolated signed-artifact qualification child. They are
//! not real-family `ShortAudioReceipt evidence.v1`, not a production output
//! plan, and not capability approval; family executors and the shared planner
//! must never consume their report as policy.
//!
//! Dual-output success never authorizes a production compact path. Marking a
//! second output can change ggml allocation and liveness enough to hide a
//! stale-output defect. Production compact authorization requires independent
//! native-only cases C and D compared against a host oracle from a separate
//! full-logits run.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlDecodeOutputPlan, GgmlDecodeReuseMode,
    GgmlGraphLifecycleCollector, GgmlGraphLifecycleSnapshot, GgmlGraphRebuildReason,
    GgmlPersistentGraphSession, RequestBackendPreference, install_graph_lifecycle_collector,
    install_request_backend_override,
};
use crate::device::execution_route::{ExecutionProvider, ResolvedExecutionRoute};

/// Bounded per-step diagnostic records on a short-audio receipt.
pub const SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS: usize = 64;

/// Receipt-facing copy of the resolved output plan. Diagnostic only.
///
/// This is a serialization projection of [`GgmlDecodeOutputPlan`]. The runtime
/// type remains the planner authority; unknown compact capability still falls
/// back uniquely to [`GgmlDecodeOutputPlan::FullLogits`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioReceiptOutputPlan {
    FullLogits,
    CompleteScores,
    NativeFirstMaxToken,
}

impl From<GgmlDecodeOutputPlan> for ShortAudioReceiptOutputPlan {
    fn from(plan: GgmlDecodeOutputPlan) -> Self {
        match plan {
            GgmlDecodeOutputPlan::FullLogits => Self::FullLogits,
            GgmlDecodeOutputPlan::CompleteScores => Self::CompleteScores,
            GgmlDecodeOutputPlan::NativeFirstMaxToken => Self::NativeFirstMaxToken,
        }
    }
}

/// Receipt-facing copy of the resolved reuse mode. Diagnostic only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortAudioReceiptReuseMode {
    FreshGraph,
    ReusableGraph,
}

impl From<GgmlDecodeReuseMode> for ShortAudioReceiptReuseMode {
    fn from(mode: GgmlDecodeReuseMode) -> Self {
        match mode {
            GgmlDecodeReuseMode::FreshGraph => Self::FreshGraph,
            GgmlDecodeReuseMode::ReusableGraph => Self::ReusableGraph,
        }
    }
}

/// Contract interpretation of a four-quadrant first divergence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeFirstDivergenceClass {
    ReusableKvOrOutputRefresh,
    SelectorOrCompactOutput,
    EncoderCrossKvOrKernel,
    PersistentCompactInteraction,
    EncoderCrossKvAllQuadrants,
    NoneObserved,
    InsufficientEvidence,
}

/// Encoder/decoder split lanes from the correctness contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncoderDecoderSplitLane {
    #[default]
    CpuEncoderCpuDecoder,
    AccelEncoderCpuDecoder,
    CpuEncoderAccelFreshDecoder,
    AccelEncoderAccelFreshDecoder,
    AccelEncoderAccelReusableDecoder,
}

/// One bounded decode step recorded on a short-audio receipt.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptDecodeStep {
    pub step: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_id: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top2_margin: Option<f32>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub graph_rebuilt: bool,
}

/// Fail-closed diagnostic block on `openasr.short-audio-receipt.v0`.
///
/// `output_plan` and `reuse_mode` are required. They record the resolved
/// [`GgmlDecodeOutputPlan`] / [`GgmlDecodeReuseMode`] projection, including the
/// unique `full_logits` fallback when compact selection is unproven.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShortAudioReceiptDecodeDiagnostics {
    pub output_plan: ShortAudioReceiptOutputPlan,
    pub reuse_mode: ShortAudioReceiptReuseMode,
    /// Planner-internal typed evidence revision that produced the immutable
    /// output/reuse decision. Legacy diagnostics may omit it; new release
    /// evidence binds it explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_evidence_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<ShortAudioReceiptDecodeStep>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_divergence: Option<DecodeFirstDivergenceClass>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub encoder_decoder_splits: Vec<EncoderDecoderSplitProbeRecord>,
}

/// One encoder/decoder split comparison record.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderDecoderSplitProbeRecord {
    pub lane: EncoderDecoderSplitLane,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub encoder_row_shape: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_checksum: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_tap_tolerance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_kv_checksum: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub step_logits_hashes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub host_token_ids: Vec<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub device_token_ids: Vec<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reusable_row_indices: Vec<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reusable_positions: Vec<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mask_hashes: Vec<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub graph_rebuilt: bool,
}

/// Contract counter-example: first-max must return token 2.
pub const DIAGNOSTIC_FIRST_MAX_TIE_LOGITS: [f32; 4] = [2.0, 1.0, 5.0, 5.0];
pub const DIAGNOSTIC_FIRST_MAX_TIE_TOKEN: i32 = 2;

/// Second row used to prove a persistent scalar output refreshes.
pub const DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS: [f32; 4] = [9.0, 1.0, 5.0, 5.0];
pub const DIAGNOSTIC_FIRST_MAX_REFRESH_TOKEN: i32 = 0;

/// Families that may enter native compact quadrants C/D. XASR, MiMo RVQ, and
/// SenseVoice stay on complete host-oracle outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticFamilyCompactPolicy {
    NativeArgmaxFirstEligible,
    LastMaxHostOracleOnly,
    FirstMaxScoreOracleOnly,
    FullFrameLogitsOnly,
}

impl DiagnosticFamilyCompactPolicy {
    pub const fn enters_native_compact_quadrants(self) -> bool {
        matches!(self, Self::NativeArgmaxFirstEligible)
    }
}

/// Decoder graph mode for one four-quadrant cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticDecoderGraphMode {
    FreshRebuild,
    ReusableGraph,
}

/// Selection mode for one four-quadrant cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticDecodeSelection {
    CompleteLogitsHostFirstMax,
    NativeArgmaxFirst,
}

/// One diagnostic dual-output execution. Agreement here is not compact
/// authorization.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticDualOutputConformanceResult {
    pub logits: Vec<f32>,
    pub device_token: i32,
    pub host_first_max_token: i32,
    pub tokens_match: bool,
    pub top2: DiagnosticTop2,
}

impl DiagnosticDualOutputConformanceResult {
    /// Dual-output agreement is diagnostic only. This never authorizes a
    /// production compact token path.
    pub const fn authorizes_production_compact(&self) -> bool {
        let _ = self;
        false
    }
}

/// Ranked top-2 finite values from one logits row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagnosticTop2 {
    pub first_index: i32,
    pub first_value: f32,
    pub second_index: Option<i32>,
    pub second_value: Option<f32>,
    pub margin: Option<f32>,
}

/// Token / receipt trace for one four-quadrant cell.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticQuadrantTrace {
    pub graph_mode: DiagnosticDecoderGraphMode,
    pub selection: DiagnosticDecodeSelection,
    pub tokens: Vec<i32>,
    pub steps: Vec<ShortAudioReceiptDecodeStep>,
    pub graph_lifecycle: GgmlGraphLifecycleSnapshot,
}

/// Four-quadrant report produced by independent exact-route runtimes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticFourQuadrantReport {
    #[serde(
        serialize_with = "serialize_execution_provider",
        deserialize_with = "deserialize_execution_provider"
    )]
    pub provider: ExecutionProvider,
    pub stable_device_id: String,
    pub vocab_size: usize,
    pub step_count: usize,
    pub case_a: DiagnosticQuadrantTrace,
    pub case_b: DiagnosticQuadrantTrace,
    pub case_c: Option<DiagnosticQuadrantTrace>,
    pub case_d: Option<DiagnosticQuadrantTrace>,
    pub classification: DecodeFirstDivergenceClass,
    pub graph_lifecycle: GgmlGraphLifecycleSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticLayer1Case {
    pub label: String,
    pub expected_tokens: Vec<i32>,
    pub actual_tokens: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticLayer1Report {
    #[serde(
        serialize_with = "serialize_execution_provider",
        deserialize_with = "deserialize_execution_provider"
    )]
    pub provider: ExecutionProvider,
    pub stable_device_id: String,
    pub cases: Vec<DiagnosticLayer1Case>,
    pub repeated_refresh_tokens: Vec<i32>,
    pub unsupported_type_rejected: bool,
    pub unsupported_layout_rejected: bool,
    pub unsupported_rank_rejected: bool,
    pub graph_lifecycle: GgmlGraphLifecycleSnapshot,
}

/// Complete Layer-1 ARGMAX_FIRST semantic fixture on one exact final route.
pub fn run_diagnostic_layer1_exact_route_probe(
    route: ResolvedExecutionRoute,
) -> Result<DiagnosticLayer1Report, GgmlCpuGraphError> {
    const FIRERED_VOCAB: usize = 7_832;
    let lifecycle = GgmlGraphLifecycleCollector::new();
    let _lifecycle_guard = install_graph_lifecycle_collector(Some(lifecycle.clone()));
    let _route_guard =
        install_request_backend_override(Some(RequestBackendPreference::Exact(route.clone())));
    let mut runner = GgmlCpuGraphRunner::new(exact_route_graph_config(&route))?;
    let mut firered = vec![0.0_f32; FIRERED_VOCAB];
    firered[100] = 1.0;
    firered[200] = 1.0;
    let cases = [
        ("unique_maximum", vec![1.0, 7.0, 3.0, 0.0], vec![1]),
        ("exact_first_tie", vec![2.0, 1.0, 5.0, 5.0], vec![2]),
        ("all_equal", vec![5.0, 5.0, 5.0, 5.0], vec![0]),
        ("negative_values", vec![-4.0, -1.0, -1.0, -8.0], vec![1]),
        (
            "leading_nan_rejected",
            vec![f32::NAN, 3.0, 3.0, 1.0],
            vec![-1],
        ),
        (
            "interior_nan_rejected",
            vec![3.0, f32::NAN, 3.0, 1.0],
            vec![-1],
        ),
        (
            "positive_infinity_rejected",
            vec![1.0, f32::INFINITY, 1.0, 0.0],
            vec![-1],
        ),
        (
            "negative_infinity_rejected",
            vec![1.0, f32::NEG_INFINITY, 1.0, 0.0],
            vec![-1],
        ),
        ("firered_vocab_width", firered, vec![100]),
        (
            "multiple_rows",
            vec![1.0, 5.0, 5.0, 2.0, 9.0, 3.0, 2.0, 1.0],
            vec![1, 0],
        ),
    ];
    let mut results = Vec::with_capacity(cases.len());
    for (label, values, expected_tokens) in cases {
        let rows = expected_tokens.len();
        let width = values.len() / rows;
        let mut graph = runner.start_graph();
        let logits = graph.new_tensor_2d_f32(width, rows, "layer1_exact_route_logits")?;
        graph.set_input(logits)?;
        let token = graph.top1_argmax_first_max(logits)?;
        graph.set_output(token)?;
        graph.set_f32_slice(logits, &values, "layer1_exact_route_logits")?;
        let actual_tokens = graph.compute_output_i32(token, rows)?;
        results.push(DiagnosticLayer1Case {
            label: label.to_string(),
            expected_tokens,
            actual_tokens,
        });
    }

    let mut repeated_refresh_tokens = Vec::with_capacity(2);
    let mut refresh = runner.start_persistent_graph_session(1024 * 1024)?;
    let (refresh_logits, refresh_token) = {
        let graph = refresh.builder();
        let logits = graph.new_tensor_2d_f32(4, 1, "layer1_refresh_logits")?;
        graph.set_input(logits)?;
        let token = graph.top1_argmax_first_max(logits)?;
        graph.set_output(token)?;
        graph.prepare_outputs_for_upload(&[token])?;
        (logits, token)
    };
    for values in [
        DIAGNOSTIC_FIRST_MAX_TIE_LOGITS,
        DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS,
    ] {
        let graph = refresh.builder();
        graph.set_f32_slice(refresh_logits, &values, "layer1_refresh_logits")?;
        repeated_refresh_tokens.extend(graph.compute_output_i32(refresh_token, 1)?);
    }
    drop(refresh);

    let graph = runner.start_graph();
    let unsupported_type =
        graph.new_tensor_2d_typed(4, 1, super::ffi::GGML_TYPE_I32, "layer1_unsupported_type")?;
    let unsupported_type_rejected = graph.top1_argmax_first_max(unsupported_type).is_err();
    drop(graph);

    let graph = runner.start_graph();
    let layout_source = graph.new_tensor_2d_f32(2, 4, "layer1_unsupported_layout")?;
    let unsupported_layout = graph.transpose(layout_source)?;
    let unsupported_layout_rejected = graph.top1_argmax_first_max(unsupported_layout).is_err();
    drop(graph);

    let graph = runner.start_graph();
    let unsupported = graph.new_tensor_3d_f32(4, 1, 2, "layer1_unsupported_rank")?;
    let unsupported_rank_rejected = graph.top1_argmax_first_max(unsupported).is_err();
    drop(graph);
    drop(runner);
    let graph_lifecycle = lifecycle.snapshot();
    validate_lifecycle_route(&graph_lifecycle, &route)?;
    if results
        .iter()
        .any(|case| case.actual_tokens != case.expected_tokens)
        || repeated_refresh_tokens
            != [
                DIAGNOSTIC_FIRST_MAX_TIE_TOKEN,
                DIAGNOSTIC_FIRST_MAX_REFRESH_TOKEN,
            ]
        || !unsupported_type_rejected
        || !unsupported_layout_rejected
        || !unsupported_rank_rejected
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "Layer-1 exact-route ARGMAX_FIRST conformance failed",
        });
    }
    Ok(DiagnosticLayer1Report {
        provider: route.provider,
        stable_device_id: route.stable_id,
        cases: results,
        repeated_refresh_tokens,
        unsupported_type_rejected,
        unsupported_layout_rejected,
        unsupported_rank_rejected,
        graph_lifecycle,
    })
}

/// Capture-aware Layer-2 producer. Capture generation remains absent unless a
/// backend API callback actually reported an executable capture generation.
pub fn run_diagnostic_layer2_exact_route_probe(
    route: ResolvedExecutionRoute,
) -> Result<DiagnosticLayer2Report, GgmlCpuGraphError> {
    let lifecycle = GgmlGraphLifecycleCollector::new();
    let _lifecycle_guard = install_graph_lifecycle_collector(Some(lifecycle.clone()));
    let _route_guard =
        install_request_backend_override(Some(RequestBackendPreference::Exact(route.clone())));
    let config = exact_route_graph_config(&route);
    let mut runner = GgmlCpuGraphRunner::new(config)?;
    let mut full_session = runner.start_persistent_graph_session(config.context_bytes)?;
    let full_logits = {
        let graph = full_session.builder();
        let logits = graph.new_tensor_2d_f32(4, 1, "layer2_full_logits")?;
        graph.set_input(logits)?;
        graph.set_output(logits)?;
        graph.prepare_outputs_for_upload(&[logits])?;
        logits
    };
    let mut full_output_refreshes = Vec::with_capacity(2);
    for values in [
        DIAGNOSTIC_FIRST_MAX_TIE_LOGITS,
        DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS,
    ] {
        let graph = full_session.builder();
        graph.set_f32_slice(full_logits, &values, "layer2_full_logits")?;
        full_output_refreshes.push(graph.compute_output_f32(full_logits, 4)?);
    }
    drop(full_session);

    let mut scalar_session = runner.rebuild_persistent_graph_session(
        config.context_bytes,
        GgmlGraphRebuildReason::TopologyChanged,
    )?;
    let (scalar_logits, scalar_token) = {
        let graph = scalar_session.builder();
        let logits = graph.new_tensor_2d_f32(4, 1, "layer2_scalar_logits")?;
        graph.set_input(logits)?;
        let token = graph.top1_argmax_first_max(logits)?;
        graph.set_output(token)?;
        graph.prepare_outputs_for_upload(&[token])?;
        (logits, token)
    };
    let mut scalar_output_refreshes = Vec::with_capacity(2);
    for values in [
        DIAGNOSTIC_FIRST_MAX_TIE_LOGITS,
        DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS,
    ] {
        let graph = scalar_session.builder();
        graph.set_f32_slice(scalar_logits, &values, "layer2_scalar_logits")?;
        scalar_output_refreshes.extend(graph.compute_output_i32(scalar_token, 1)?);
    }
    drop(scalar_session);

    let mut arena = runner.start_state_tensor_arena(config.context_bytes)?;
    let cache = arena.new_tensor_2d_typed(4, 4, super::ffi::GGML_TYPE_F32, "layer2_kv")?;
    arena.set_f32_slice(cache, &[0.0; 16], "layer2_kv")?;
    let mut state_session = runner.rebuild_persistent_graph_session(
        config.context_bytes,
        GgmlGraphRebuildReason::TopologyChanged,
    )?;
    let (token_embeddings, position_embeddings, token_id, position_id, mask, row, state_output) = {
        let graph = state_session.builder();
        let token_embeddings = graph.new_tensor_2d_f32(4, 2, "layer2_token_embeddings")?;
        let position_embeddings = graph.new_tensor_2d_f32(4, 2, "layer2_position_embeddings")?;
        let token_id = graph.new_tensor_1d_i32(1, "layer2_token_id")?;
        let position_id = graph.new_tensor_1d_i32(1, "layer2_position_id")?;
        let mask = graph.new_tensor_1d_f32(4, "layer2_mask")?;
        let row = graph.new_tensor_1d_i32(1, "layer2_kv_row")?;
        for input in [
            token_embeddings,
            position_embeddings,
            token_id,
            position_id,
            mask,
            row,
        ] {
            graph.set_input(input)?;
        }
        let token_state = graph.get_rows(token_embeddings, token_id)?;
        let position_state = graph.get_rows(position_embeddings, position_id)?;
        let state = graph.add(token_state, position_state)?;
        let state = graph.add(state, mask)?;
        let state_output = graph.set_kv_rows(arena.graph_tensor(cache), state, row)?;
        graph.add_kv_write_root(state_output)?;
        graph.set_output(state_output)?;
        graph.prepare_outputs_for_upload(&[state_output])?;
        (
            token_embeddings,
            position_embeddings,
            token_id,
            position_id,
            mask,
            row,
            state_output,
        )
    };
    {
        let graph = state_session.builder();
        graph.set_f32_slice(
            token_embeddings,
            &[2.0, 1.0, 5.0, 5.0, 9.0, 1.0, 5.0, 5.0],
            "layer2_token_embeddings",
        )?;
        graph.set_f32_slice(
            position_embeddings,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "layer2_position_embeddings",
        )?;
    }
    let mut kv_state_refreshes = Vec::with_capacity(2);
    for (token, position, mask_values, row_index) in [
        (0_i32, 0_i32, [0.0_f32, 0.0, 0.0, 0.0], 0_i32),
        (1_i32, 1_i32, [-10.0_f32, 0.0, 0.0, 2.0], 1_i32),
    ] {
        let graph = state_session.builder();
        graph.set_i32_slice(token_id, &[token], "layer2_token_id")?;
        graph.set_i32_slice(position_id, &[position], "layer2_position_id")?;
        graph.set_f32_slice(mask, &mask_values, "layer2_mask")?;
        graph.set_i32_slice(row, &[row_index], "layer2_kv_row")?;
        kv_state_refreshes.push(graph.compute_output_f32(state_output, 16)?);
    }
    state_session.mark_poisoned_after_failed_compute();
    let poisoned_reexecution_rejected = matches!(
        state_session.builder().compute_output_f32(state_output, 16),
        Err(GgmlCpuGraphError::GraphSessionPoisoned)
    );
    drop(state_session);
    drop(arena);
    drop(runner);

    let expected_kv_first = [
        2.0_f32, 1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let expected_kv_second = [
        2.0_f32, 1.0, 5.0, 5.0, -1.0, 2.0, 5.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    if full_output_refreshes
        != vec![
            DIAGNOSTIC_FIRST_MAX_TIE_LOGITS.to_vec(),
            DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS.to_vec(),
        ]
        || scalar_output_refreshes
            != [
                DIAGNOSTIC_FIRST_MAX_TIE_TOKEN,
                DIAGNOSTIC_FIRST_MAX_REFRESH_TOKEN,
            ]
        || kv_state_refreshes != vec![expected_kv_first.to_vec(), expected_kv_second.to_vec()]
        || !poisoned_reexecution_rejected
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "Layer-2 exact-route refresh or poison conformance failed",
        });
    }

    let graph_lifecycle = lifecycle.snapshot();
    validate_lifecycle_route(&graph_lifecycle, &route)?;
    validate_layer2_lifecycle(&graph_lifecycle, route.provider)?;
    Ok(DiagnosticLayer2Report {
        provider: route.provider,
        stable_device_id: route.stable_id,
        full_output_refreshes,
        scalar_output_refreshes,
        kv_state_refreshes,
        poisoned_reexecution_rejected,
        graph_lifecycle,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticLayer2Report {
    #[serde(
        serialize_with = "serialize_execution_provider",
        deserialize_with = "deserialize_execution_provider"
    )]
    pub provider: ExecutionProvider,
    pub stable_device_id: String,
    pub full_output_refreshes: Vec<Vec<f32>>,
    pub scalar_output_refreshes: Vec<i32>,
    pub kv_state_refreshes: Vec<Vec<f32>>,
    pub poisoned_reexecution_rejected: bool,
    pub graph_lifecycle: GgmlGraphLifecycleSnapshot,
}

/// Inputs used to classify a four-quadrant first divergence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticFourQuadrantClassificationInput<'a> {
    pub case_a: &'a [i32],
    pub case_b: &'a [i32],
    pub case_c: Option<&'a [i32]>,
    pub case_d: Option<&'a [i32]>,
    pub cpu_reference: Option<&'a [i32]>,
}

/// First FireRed encoder kernel-stage fork between two checksumed runs.
///
/// Intra-block order is `ffn1_out` -> `attn_out` (relative-position attention)
/// -> `conv_out` (depthwise) -> `ffn2_out` -> `block_out`. This enum is the
/// contract split. A fully bounded subsample-stem tap sequence names the first
/// graph seam before the Conformer blocks; incomplete evidence still collapses
/// to [`Self::InsufficientEvidence`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncoderKernelStageClass {
    SubsampleInput,
    SubsampleConvolution,
    SubsampleBias,
    SubsampleRelu,
    SubsampleLayout,
    SubsampleOutputProjection,
    RelativePositionAttention,
    DepthwiseConvolution,
    EncoderReadback,
    InsufficientEvidence,
}

/// Bit-exact SHA-256 pair (`diagnostic_logits_sha256` hex) for one tap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderKernelStageChecksumPair<'a> {
    pub cpu: &'a str,
    pub accel: &'a str,
}

impl EncoderKernelStageChecksumPair<'_> {
    fn differs(self) -> bool {
        self.cpu != self.accel
    }
}

/// Per-layer checksums for encoder kernel-stage classification.
///
/// Values are hex SHA-256 strings, not family tensor types, so the classifier
/// can sit in the shared diagnostic module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderKernelStageLayerChecksums<'a> {
    pub layer_index: usize,
    pub ffn1_out: Option<EncoderKernelStageChecksumPair<'a>>,
    pub attn_out: Option<EncoderKernelStageChecksumPair<'a>>,
    pub conv_out: Option<EncoderKernelStageChecksumPair<'a>>,
    pub ffn2_out: Option<EncoderKernelStageChecksumPair<'a>>,
    pub block_out: Option<EncoderKernelStageChecksumPair<'a>>,
}

/// Ordered test-only FireRed Conv2d-subsampling checkpoints.
///
/// A classifier may name a subsample seam only when every checkpoint through
/// that seam is present. This prevents a partial diagnostic dump from being
/// promoted to an operator-level conclusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderKernelStageStemChecksums<'a> {
    pub mel_4d: Option<EncoderKernelStageChecksumPair<'a>>,
    pub conv1_raw: Option<EncoderKernelStageChecksumPair<'a>>,
    pub conv1_bias: Option<EncoderKernelStageChecksumPair<'a>>,
    pub conv1_relu: Option<EncoderKernelStageChecksumPair<'a>>,
    pub conv2_raw: Option<EncoderKernelStageChecksumPair<'a>>,
    pub conv2_bias: Option<EncoderKernelStageChecksumPair<'a>>,
    pub conv2_relu: Option<EncoderKernelStageChecksumPair<'a>>,
    pub after_permute: Option<EncoderKernelStageChecksumPair<'a>>,
    pub after_cont: Option<EncoderKernelStageChecksumPair<'a>>,
    pub flat_2d: Option<EncoderKernelStageChecksumPair<'a>>,
    pub out_matmul: Option<EncoderKernelStageChecksumPair<'a>>,
    pub subsample_out: Option<EncoderKernelStageChecksumPair<'a>>,
}

/// Checksum-string input for [`classify_encoder_kernel_stage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderKernelStageClassificationInput<'a> {
    pub stem: Option<EncoderKernelStageStemChecksums<'a>>,
    pub subsample: Option<EncoderKernelStageChecksumPair<'a>>,
    pub layers: &'a [EncoderKernelStageLayerChecksums<'a>],
    pub encoder_output: Option<EncoderKernelStageChecksumPair<'a>>,
}

/// Class plus the first diverging layer/tap, when the evidence names one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderKernelStageClassification {
    pub class: EncoderKernelStageClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_divergent_layer: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_divergent_tap: Option<String>,
}

fn encoder_kernel_stage_result(
    class: EncoderKernelStageClass,
    layer: Option<usize>,
    tap: Option<&'static str>,
) -> EncoderKernelStageClassification {
    EncoderKernelStageClassification {
        class,
        first_divergent_layer: layer,
        first_divergent_tap: tap.map(str::to_string),
    }
}

/// Classify the first CPU-vs-accelerator encoder kernel-stage divergence.
///
/// Checksums are opaque hex strings; the caller must hash with the same
/// bit-exact `f32::to_bits` little-endian SHA-256 as
/// [`diagnostic_logits_sha256`].
pub fn classify_encoder_kernel_stage(
    input: EncoderKernelStageClassificationInput<'_>,
) -> EncoderKernelStageClassification {
    if let Some(stem) = input.stem {
        let ordered = [
            (
                stem.mel_4d,
                EncoderKernelStageClass::SubsampleInput,
                "mel_4d",
            ),
            (
                stem.conv1_raw,
                EncoderKernelStageClass::SubsampleConvolution,
                "conv1_raw",
            ),
            (
                stem.conv1_bias,
                EncoderKernelStageClass::SubsampleBias,
                "conv1_bias",
            ),
            (
                stem.conv1_relu,
                EncoderKernelStageClass::SubsampleRelu,
                "conv1_relu",
            ),
            (
                stem.conv2_raw,
                EncoderKernelStageClass::SubsampleConvolution,
                "conv2_raw",
            ),
            (
                stem.conv2_bias,
                EncoderKernelStageClass::SubsampleBias,
                "conv2_bias",
            ),
            (
                stem.conv2_relu,
                EncoderKernelStageClass::SubsampleRelu,
                "conv2_relu",
            ),
            (
                stem.after_permute,
                EncoderKernelStageClass::SubsampleLayout,
                "after_permute",
            ),
            (
                stem.after_cont,
                EncoderKernelStageClass::SubsampleLayout,
                "after_cont",
            ),
            (
                stem.flat_2d,
                EncoderKernelStageClass::SubsampleLayout,
                "flat_2d",
            ),
            (
                stem.out_matmul,
                EncoderKernelStageClass::SubsampleOutputProjection,
                "out_matmul",
            ),
            (
                stem.subsample_out,
                EncoderKernelStageClass::SubsampleOutputProjection,
                "subsample_out",
            ),
        ];
        for (tap, class, name) in ordered {
            let Some(tap) = tap else {
                return encoder_kernel_stage_result(
                    EncoderKernelStageClass::InsufficientEvidence,
                    None,
                    Some(name),
                );
            };
            if tap.differs() {
                return encoder_kernel_stage_result(class, None, Some(name));
            }
        }
    }

    if input
        .subsample
        .is_some_and(EncoderKernelStageChecksumPair::differs)
    {
        return encoder_kernel_stage_result(
            EncoderKernelStageClass::InsufficientEvidence,
            None,
            Some("subsample_out"),
        );
    }

    let mut first_block_divergence: Option<&EncoderKernelStageLayerChecksums<'_>> = None;
    for layer in input.layers {
        if let Some(block_out) = layer.block_out
            && block_out.differs()
        {
            first_block_divergence = Some(layer);
            break;
        }
    }

    if let Some(layer) = first_block_divergence {
        let Some(ffn1) = layer.ffn1_out else {
            return encoder_kernel_stage_result(
                EncoderKernelStageClass::InsufficientEvidence,
                Some(layer.layer_index),
                Some("block_out"),
            );
        };
        let Some(attn) = layer.attn_out else {
            return encoder_kernel_stage_result(
                EncoderKernelStageClass::InsufficientEvidence,
                Some(layer.layer_index),
                Some("block_out"),
            );
        };
        let Some(conv) = layer.conv_out else {
            return encoder_kernel_stage_result(
                EncoderKernelStageClass::InsufficientEvidence,
                Some(layer.layer_index),
                Some("block_out"),
            );
        };
        let Some(ffn2) = layer.ffn2_out else {
            return encoder_kernel_stage_result(
                EncoderKernelStageClass::InsufficientEvidence,
                Some(layer.layer_index),
                Some("block_out"),
            );
        };

        if ffn1.differs() {
            return encoder_kernel_stage_result(
                EncoderKernelStageClass::InsufficientEvidence,
                Some(layer.layer_index),
                Some("ffn1_out"),
            );
        }
        if attn.differs() {
            return encoder_kernel_stage_result(
                EncoderKernelStageClass::RelativePositionAttention,
                Some(layer.layer_index),
                Some("attn_out"),
            );
        }
        if conv.differs() {
            return encoder_kernel_stage_result(
                EncoderKernelStageClass::DepthwiseConvolution,
                Some(layer.layer_index),
                Some("conv_out"),
            );
        }
        if ffn2.differs() {
            return encoder_kernel_stage_result(
                EncoderKernelStageClass::InsufficientEvidence,
                Some(layer.layer_index),
                Some("ffn2_out"),
            );
        }
        let tap = if layer
            .block_out
            .is_some_and(EncoderKernelStageChecksumPair::differs)
        {
            "block_out"
        } else {
            "encoder_output"
        };
        return encoder_kernel_stage_result(
            EncoderKernelStageClass::EncoderReadback,
            Some(layer.layer_index),
            Some(tap),
        );
    }

    if input
        .encoder_output
        .is_some_and(EncoderKernelStageChecksumPair::differs)
    {
        return encoder_kernel_stage_result(
            EncoderKernelStageClass::EncoderReadback,
            None,
            Some("encoder_output"),
        );
    }

    encoder_kernel_stage_result(EncoderKernelStageClass::InsufficientEvidence, None, None)
}

/// Host first-max oracle: the lowest finite index wins exact ties.
pub fn diagnostic_host_first_max_token(logits: &[f32]) -> Option<i32> {
    let mut best_index = None;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if best_index.is_none() || value > best_value {
            best_index = Some(i32::try_from(index).ok()?);
            best_value = value;
        }
    }
    best_index
}

/// Host last-max oracle used only to document XASR's current family policy.
pub fn diagnostic_host_last_max_token(logits: &[f32]) -> Option<i32> {
    logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .and_then(|(index, _)| i32::try_from(index).ok())
}

pub fn diagnostic_top2(logits: &[f32]) -> Option<DiagnosticTop2> {
    let mut ranked = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(index, value)| (index, *value))
        .collect::<Vec<_>>();
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    let (first_index, first_value) = ranked[0];
    let second = ranked.get(1).copied();
    Some(DiagnosticTop2 {
        first_index: i32::try_from(first_index).ok()?,
        first_value,
        second_index: second.and_then(|(index, _)| i32::try_from(index).ok()),
        second_value: second.map(|(_, value)| value),
        margin: second.map(|(_, value)| first_value - value),
    })
}

pub fn diagnostic_logits_sha256(logits: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(logits.len().saturating_mul(4));
    for value in logits {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    hex_lower(Sha256::digest(bytes))
}

fn hex_lower(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Same-graph dual-output diagnostic: one logits producer, two marked outputs.
///
/// The returned agreement must not be treated as production compact evidence.
pub fn run_diagnostic_dual_output_conformance(
    rows: &[&[f32]],
) -> Result<Vec<DiagnosticDualOutputConformanceResult>, GgmlCpuGraphError> {
    if rows.is_empty() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "diagnostic dual-output requires at least one logits row",
        });
    }
    let vocab = rows[0].len();
    if vocab == 0 || rows.iter().any(|row| row.len() != vocab) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "diagnostic dual-output rows must share a non-empty vocab width",
        });
    }

    let mut runner = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())?;
    let mut session = runner.start_persistent_graph_session(1024 * 1024)?;
    let (logits, token) = build_diagnostic_dual_output_graph(session.builder(), vocab)?;

    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        session
            .builder()
            .set_f32_slice(logits, row, "diagnostic_dual_output_logits")?;
        let (logits_out, tokens_out) = session
            .builder()
            .compute_outputs_f32_i32(&[(logits, vocab)], &[(token, 1)])?;
        results.push(diagnostic_dual_output_result(
            &logits_out[0],
            tokens_out[0][0],
        )?);
    }
    Ok(results)
}

fn build_diagnostic_dual_output_graph<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    vocab: usize,
) -> Result<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>), GgmlCpuGraphError> {
    let logits = graph.new_tensor_2d_f32(vocab, 1, "diagnostic_dual_output_logits")?;
    graph.set_input(logits)?;
    graph.set_output(logits)?;
    let token = graph.top1_argmax_first_max(logits)?;
    graph.set_output(token)?;
    graph.prepare_outputs_for_upload(&[logits, token])?;
    Ok((logits, token))
}

fn diagnostic_dual_output_result(
    logits: &[f32],
    device_token: i32,
) -> Result<DiagnosticDualOutputConformanceResult, GgmlCpuGraphError> {
    let host_first_max_token =
        diagnostic_host_first_max_token(logits).ok_or(GgmlCpuGraphError::UnsupportedInputs {
            reason: "diagnostic dual-output host first-max found no finite logit",
        })?;
    let top2 = diagnostic_top2(logits).ok_or(GgmlCpuGraphError::UnsupportedInputs {
        reason: "diagnostic dual-output top-2 found no finite logit",
    })?;
    Ok(DiagnosticDualOutputConformanceResult {
        logits: logits.to_vec(),
        device_token,
        host_first_max_token,
        tokens_match: device_token == host_first_max_token,
        top2,
    })
}

/// Fresh/reuse four-quadrant probe on bounded logits fixtures.
///
/// Every quadrant owns an independent runtime instance so neither KV/storage
/// state nor output topology can leak between complete and compact cases. Cases
/// C/D are omitted when the family must stay on a host oracle (XASR last-max,
/// MiMo RVQ first-max scores, SenseVoice full frames).
pub fn run_diagnostic_four_quadrant_cpu_probe(
    steps: &[&[f32]],
    family_policy: DiagnosticFamilyCompactPolicy,
) -> Result<DiagnosticFourQuadrantReport, GgmlCpuGraphError> {
    run_diagnostic_four_quadrant_exact_route_probe(
        ResolvedExecutionRoute::cpu(),
        steps,
        family_policy,
    )
}

pub const GPU_DECODE_CONFORMANCE_SCHEMA: &str = "openasr.gpu-decode-conformance.v1";
pub const GPU_DECODE_CONFORMANCE_PRODUCTION_VOCAB: usize = 7_832;

/// Strict backend-neutral operator/lifecycle suite run by the isolated
/// qualification child after its signed final provider has been loaded. This
/// remains Layer 1/2 evidence: it cannot substitute for a real-family
/// `ShortAudioReceipt evidence.v1` row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticDecodeConformanceSuite {
    pub schema: String,
    pub result: String,
    #[serde(
        serialize_with = "serialize_execution_provider",
        deserialize_with = "deserialize_execution_provider"
    )]
    pub provider: ExecutionProvider,
    pub stable_device_id: String,
    pub layer1: DiagnosticLayer1Report,
    pub layer2: DiagnosticLayer2Report,
    pub production_shape_four_quadrant: DiagnosticFourQuadrantReport,
}

impl DiagnosticDecodeConformanceSuite {
    /// Revalidate a serialized qualification-child result before a parent or
    /// later evidence envelope accepts it. This is data validation only; it
    /// never turns the diagnostic suite into capability approval.
    pub fn validate(&self) -> Result<(), GgmlCpuGraphError> {
        let expected_cases = [
            ("unique_maximum", vec![1]),
            ("exact_first_tie", vec![2]),
            ("all_equal", vec![0]),
            ("negative_values", vec![1]),
            ("leading_nan_rejected", vec![-1]),
            ("interior_nan_rejected", vec![-1]),
            ("positive_infinity_rejected", vec![-1]),
            ("negative_infinity_rejected", vec![-1]),
            ("firered_vocab_width", vec![100]),
            ("multiple_rows", vec![1, 0]),
        ];
        let expected_quadrant_tokens = [100, 0, 0, 4_000];
        let layer1_valid =
            self.layer1.provider == self.provider
                && self.layer1.stable_device_id == self.stable_device_id
                && self.layer1.cases.len() == expected_cases.len()
                && self.layer1.cases.iter().zip(expected_cases).all(
                    |(actual, (label, expected))| {
                        actual.label == label
                            && actual.expected_tokens == expected
                            && actual.actual_tokens == expected
                    },
                )
                && self.layer1.repeated_refresh_tokens
                    == [
                        DIAGNOSTIC_FIRST_MAX_TIE_TOKEN,
                        DIAGNOSTIC_FIRST_MAX_REFRESH_TOKEN,
                    ]
                && self.layer1.unsupported_type_rejected
                && self.layer1.unsupported_layout_rejected
                && self.layer1.unsupported_rank_rejected;
        let layer2_valid = self.layer2.provider == self.provider
            && self.layer2.stable_device_id == self.stable_device_id
            && self.layer2.full_output_refreshes
                == vec![
                    DIAGNOSTIC_FIRST_MAX_TIE_LOGITS.to_vec(),
                    DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS.to_vec(),
                ]
            && self.layer2.scalar_output_refreshes
                == [
                    DIAGNOSTIC_FIRST_MAX_TIE_TOKEN,
                    DIAGNOSTIC_FIRST_MAX_REFRESH_TOKEN,
                ]
            && self.layer2.kv_state_refreshes
                == vec![
                    vec![
                        2.0, 1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                    vec![
                        2.0, 1.0, 5.0, 5.0, -1.0, 2.0, 5.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0,
                    ],
                ]
            && self.layer2.poisoned_reexecution_rejected;
        let quadrants = &self.production_shape_four_quadrant;
        let quadrant_identity_valid = quadrants.provider == self.provider
            && quadrants.stable_device_id == self.stable_device_id
            && quadrants.vocab_size == GPU_DECODE_CONFORMANCE_PRODUCTION_VOCAB
            && quadrants.step_count == expected_quadrant_tokens.len()
            && quadrants.classification == DecodeFirstDivergenceClass::NoneObserved;
        let expected_quadrants = [
            (
                Some(&quadrants.case_a),
                DiagnosticDecoderGraphMode::FreshRebuild,
                DiagnosticDecodeSelection::CompleteLogitsHostFirstMax,
            ),
            (
                Some(&quadrants.case_b),
                DiagnosticDecoderGraphMode::ReusableGraph,
                DiagnosticDecodeSelection::CompleteLogitsHostFirstMax,
            ),
            (
                quadrants.case_c.as_ref(),
                DiagnosticDecoderGraphMode::FreshRebuild,
                DiagnosticDecodeSelection::NativeArgmaxFirst,
            ),
            (
                quadrants.case_d.as_ref(),
                DiagnosticDecoderGraphMode::ReusableGraph,
                DiagnosticDecodeSelection::NativeArgmaxFirst,
            ),
        ];
        let quadrant_traces_valid =
            expected_quadrants
                .iter()
                .all(|(trace, graph_mode, selection)| {
                    trace.is_some_and(|trace| {
                        trace.graph_mode == *graph_mode
                            && trace.selection == *selection
                            && trace.tokens == expected_quadrant_tokens
                            && trace.steps.len() == expected_quadrant_tokens.len()
                            && trace
                                .steps
                                .iter()
                                .zip(expected_quadrant_tokens)
                                .enumerate()
                                .all(|(step, (record, token))| {
                                    record.step == step as u32
                                        && record.token_id == Some(token)
                                        && record.graph_rebuilt
                                            == matches!(
                                                graph_mode,
                                                DiagnosticDecoderGraphMode::FreshRebuild
                                            )
                                })
                            && validate_quadrant_lifecycle(trace).is_ok()
                    })
                });
        let combined_quadrant_events = expected_quadrants
            .iter()
            .filter_map(|(trace, ..)| *trace)
            .flat_map(|trace| trace.graph_lifecycle.events.iter())
            .collect::<Vec<_>>();
        let full_quadrant_events = quadrants.graph_lifecycle.events.iter().collect::<Vec<_>>();
        if self.schema != GPU_DECODE_CONFORMANCE_SCHEMA
            || self.result != "pass"
            || self.stable_device_id.trim().is_empty()
            || self.stable_device_id.len() > 128
            || !layer1_valid
            || !layer2_valid
            || !quadrant_identity_valid
            || !quadrant_traces_valid
            || combined_quadrant_events != full_quadrant_events
            || validate_lifecycle_identity(
                &self.layer1.graph_lifecycle,
                self.provider,
                &self.stable_device_id,
            )
            .is_err()
            || summarize_lifecycle(&self.layer1.graph_lifecycle).is_err()
            || validate_lifecycle_identity(
                &self.layer2.graph_lifecycle,
                self.provider,
                &self.stable_device_id,
            )
            .is_err()
            || validate_layer2_lifecycle(&self.layer2.graph_lifecycle, self.provider).is_err()
            || validate_lifecycle_identity(
                &quadrants.graph_lifecycle,
                self.provider,
                &self.stable_device_id,
            )
            .is_err()
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "serialized decode conformance suite is incomplete or inconsistent",
            });
        }
        Ok(())
    }
}

pub fn run_diagnostic_decode_conformance_suite(
    route: ResolvedExecutionRoute,
) -> Result<DiagnosticDecodeConformanceSuite, GgmlCpuGraphError> {
    let layer1 = run_diagnostic_layer1_exact_route_probe(route.clone())?;
    let layer2 = run_diagnostic_layer2_exact_route_probe(route.clone())?;
    let rows = production_shape_first_max_rows();
    let row_refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let four_quadrant = run_diagnostic_four_quadrant_exact_route_probe(
        route.clone(),
        &row_refs,
        DiagnosticFamilyCompactPolicy::NativeArgmaxFirstEligible,
    )?;
    if four_quadrant.classification != DecodeFirstDivergenceClass::NoneObserved
        || four_quadrant.case_c.is_none()
        || four_quadrant.case_d.is_none()
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "production-shape four-quadrant conformance diverged",
        });
    }
    let suite = DiagnosticDecodeConformanceSuite {
        schema: GPU_DECODE_CONFORMANCE_SCHEMA.to_string(),
        result: "pass".to_string(),
        provider: route.provider,
        stable_device_id: route.stable_id,
        layer1,
        layer2,
        production_shape_four_quadrant: four_quadrant,
    };
    suite.validate()?;
    Ok(suite)
}

fn production_shape_first_max_rows() -> Vec<Vec<f32>> {
    let mut tie = vec![-8.0_f32; GPU_DECODE_CONFORMANCE_PRODUCTION_VOCAB];
    tie[100] = 5.0;
    tie[200] = 5.0;
    let mut changed = vec![-9.0_f32; GPU_DECODE_CONFORMANCE_PRODUCTION_VOCAB];
    changed[0] = 7.0;
    let equal = vec![0.0_f32; GPU_DECODE_CONFORMANCE_PRODUCTION_VOCAB];
    let mut negative_tie = vec![-12.0_f32; GPU_DECODE_CONFORMANCE_PRODUCTION_VOCAB];
    negative_tie[4_000] = -1.0;
    negative_tie[4_001] = -1.0;
    vec![tie, changed, equal, negative_tie]
}

/// Backend-neutral four-quadrant producer bound to one exact live ggml route.
/// The runner re-attests the initialized device on every compute; lifecycle
/// events are rejected if they name a different final provider/device.
pub fn run_diagnostic_four_quadrant_exact_route_probe(
    route: ResolvedExecutionRoute,
    steps: &[&[f32]],
    family_policy: DiagnosticFamilyCompactPolicy,
) -> Result<DiagnosticFourQuadrantReport, GgmlCpuGraphError> {
    if steps.is_empty() || steps.len() > SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "diagnostic four-quadrant step count is empty or unbounded",
        });
    }
    let vocab = steps[0].len();
    if vocab == 0 || steps.iter().any(|row| row.len() != vocab) {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "diagnostic four-quadrant rows must share a non-empty vocab width",
        });
    }

    let lifecycle = GgmlGraphLifecycleCollector::new();
    let _lifecycle_guard = install_graph_lifecycle_collector(Some(lifecycle.clone()));
    let _route_guard =
        install_request_backend_override(Some(RequestBackendPreference::Exact(route.clone())));
    let config = exact_route_graph_config(&route);
    let case_a = {
        let checkpoint = lifecycle.checkpoint();
        let mut runtime = GgmlCpuGraphRunner::new(config)?;
        let trace = run_fresh_complete_logits(&mut runtime, steps)?;
        drop(runtime);
        bind_quadrant_lifecycle(trace, &lifecycle, checkpoint, &route)?
    };
    let case_b = {
        let checkpoint = lifecycle.checkpoint();
        let mut runtime = GgmlCpuGraphRunner::new(config)?;
        let trace = run_reusable_complete_logits(&mut runtime, steps)?;
        drop(runtime);
        bind_quadrant_lifecycle(trace, &lifecycle, checkpoint, &route)?
    };
    let (case_c, case_d) = if family_policy.enters_native_compact_quadrants() {
        let case_c = {
            let checkpoint = lifecycle.checkpoint();
            let mut runtime = GgmlCpuGraphRunner::new(config)?;
            let trace = run_fresh_native_argmax(&mut runtime, steps)?;
            drop(runtime);
            bind_quadrant_lifecycle(trace, &lifecycle, checkpoint, &route)?
        };
        let case_d = {
            let checkpoint = lifecycle.checkpoint();
            let mut runtime = GgmlCpuGraphRunner::new(config)?;
            let trace = run_reusable_native_argmax(&mut runtime, steps)?;
            drop(runtime);
            bind_quadrant_lifecycle(trace, &lifecycle, checkpoint, &route)?
        };
        (Some(case_c), Some(case_d))
    } else {
        (None, None)
    };

    let cpu_reference = steps
        .iter()
        .map(|row| {
            diagnostic_host_first_max_token(row).ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "diagnostic four-quadrant host first-max found no finite logit",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let classification =
        classify_four_quadrant_first_divergence(DiagnosticFourQuadrantClassificationInput {
            case_a: &case_a.tokens,
            case_b: &case_b.tokens,
            case_c: case_c.as_ref().map(|trace| trace.tokens.as_slice()),
            case_d: case_d.as_ref().map(|trace| trace.tokens.as_slice()),
            cpu_reference: Some(&cpu_reference),
        });

    let graph_lifecycle = lifecycle.snapshot();
    validate_lifecycle_route(&graph_lifecycle, &route)?;
    Ok(DiagnosticFourQuadrantReport {
        provider: route.provider,
        stable_device_id: route.stable_id,
        vocab_size: vocab,
        step_count: steps.len(),
        case_a,
        case_b,
        case_c,
        case_d,
        classification,
        graph_lifecycle,
    })
}

fn exact_route_graph_config(route: &ResolvedExecutionRoute) -> GgmlCpuGraphConfig {
    let backend = match route.provider {
        ExecutionProvider::Cpu => GgmlCpuGraphBackend::Cpu,
        ExecutionProvider::Metal => GgmlCpuGraphBackend::Metal,
        ExecutionProvider::Cuda
        | ExecutionProvider::Hip
        | ExecutionProvider::Vulkan
        | ExecutionProvider::Accelerator
        | ExecutionProvider::Unknown => GgmlCpuGraphBackend::Gpu,
    };
    let mut config = GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend);
    // Formal FullDevice conformance must exercise the direct selected backend.
    // An ambient scheduler env cannot silently turn it into a multi-backend or
    // CPU-fallback probe, and capture evidence is observable only on this route.
    config.use_scheduler = false;
    config
}

fn serialize_execution_provider<S>(
    provider: &ExecutionProvider,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(provider.as_str())
}

fn deserialize_execution_provider<'de, D>(deserializer: D) -> Result<ExecutionProvider, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    match value.as_str() {
        "cpu" => Ok(ExecutionProvider::Cpu),
        "metal" => Ok(ExecutionProvider::Metal),
        "cuda" => Ok(ExecutionProvider::Cuda),
        "hip" => Ok(ExecutionProvider::Hip),
        "vulkan" => Ok(ExecutionProvider::Vulkan),
        "accelerator" => Ok(ExecutionProvider::Accelerator),
        "unknown" => Ok(ExecutionProvider::Unknown),
        _ => Err(serde::de::Error::unknown_variant(
            &value,
            &[
                "cpu",
                "metal",
                "cuda",
                "hip",
                "vulkan",
                "accelerator",
                "unknown",
            ],
        )),
    }
}

fn bind_quadrant_lifecycle(
    mut trace: DiagnosticQuadrantTrace,
    collector: &GgmlGraphLifecycleCollector,
    checkpoint: (usize, bool),
    route: &ResolvedExecutionRoute,
) -> Result<DiagnosticQuadrantTrace, GgmlCpuGraphError> {
    let snapshot = collector.snapshot();
    if snapshot.overflowed || checkpoint.1 || checkpoint.0 >= snapshot.events.len() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "diagnostic quadrant lifecycle is empty or overflowed",
        });
    }
    trace.graph_lifecycle = GgmlGraphLifecycleSnapshot {
        events: snapshot.events[checkpoint.0..].to_vec(),
        overflowed: false,
    };
    validate_lifecycle_route(&trace.graph_lifecycle, route)?;
    validate_quadrant_lifecycle(&trace)?;
    Ok(trace)
}

fn validate_lifecycle_route(
    lifecycle: &GgmlGraphLifecycleSnapshot,
    route: &ResolvedExecutionRoute,
) -> Result<(), GgmlCpuGraphError> {
    validate_lifecycle_identity(lifecycle, route.provider, &route.stable_id)
}

fn validate_lifecycle_identity(
    lifecycle: &GgmlGraphLifecycleSnapshot,
    provider: ExecutionProvider,
    stable_device_id: &str,
) -> Result<(), GgmlCpuGraphError> {
    if lifecycle.overflowed
        || lifecycle.events.is_empty()
        || lifecycle
            .events
            .iter()
            .any(|event| event.provider != provider.as_str() || event.device != stable_device_id)
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "diagnostic lifecycle is empty, overflowed, or not bound to the exact route",
        });
    }
    Ok(())
}

#[derive(Default)]
struct DiagnosticGraphLifecycleStats {
    scheduler_enabled: Option<bool>,
    prepare_generation: Option<u64>,
    input_generations: BTreeSet<u64>,
    computes_started: Vec<(u64, Option<u64>, Option<u64>, Option<u64>)>,
    computes_completed: BTreeMap<u64, u64>,
    output_reads: Vec<(u64, u64)>,
    kv_commits: BTreeSet<u64>,
    capture_before: usize,
    capture_after: usize,
    capture_before_pending: bool,
    capture_before_computes: BTreeSet<u64>,
    capture_after_computes: BTreeSet<u64>,
    active_compute: Option<u64>,
    capture_supported: Option<bool>,
    graph_tracked: Option<bool>,
    capture_enabled: Option<bool>,
    capture_executable_present: bool,
    capture_generation: Option<u64>,
    last_capture_phase: Option<super::GgmlCaptureObservationPhase>,
    capture_enabled_observed: bool,
    capture_executable_observed: bool,
    poisoned_sequence: Option<u64>,
    dropped_sequence: Option<u64>,
}

struct DiagnosticLifecycleSummary {
    graphs: BTreeMap<(u64, u64), DiagnosticGraphLifecycleStats>,
    rebuilds: Vec<(u64, u64, GgmlGraphRebuildReason)>,
}

fn summarize_lifecycle(
    lifecycle: &GgmlGraphLifecycleSnapshot,
) -> Result<DiagnosticLifecycleSummary, GgmlCpuGraphError> {
    if lifecycle.overflowed || lifecycle.events.is_empty() {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "conformance lifecycle is empty or overflowed",
        });
    }
    let mut graphs = BTreeMap::<(u64, u64), DiagnosticGraphLifecycleStats>::new();
    let mut graph_instances = BTreeSet::new();
    let mut graph_generations = BTreeSet::new();
    let mut rebuilds = Vec::new();
    let mut previous_sequence = 0_u64;
    for event in &lifecycle.events {
        if event.sequence <= previous_sequence {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "conformance lifecycle sequence is not strictly increasing",
            });
        }
        previous_sequence = event.sequence;
        let key = (event.graph_instance, event.graph_generation);
        match &event.kind {
            super::GgmlGraphLifecycleEventKind::Created { scheduler_enabled }
            | super::GgmlGraphLifecycleEventKind::ExistingGraphObserved {
                scheduler_enabled, ..
            } => {
                if graphs.contains_key(&key)
                    || !graph_instances.insert(event.graph_instance)
                    || !graph_generations.insert(event.graph_generation)
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance lifecycle attaches one graph generation twice",
                    });
                }
                graphs.insert(
                    key,
                    DiagnosticGraphLifecycleStats {
                        scheduler_enabled: Some(*scheduler_enabled),
                        prepare_generation: match &event.kind {
                            super::GgmlGraphLifecycleEventKind::ExistingGraphObserved {
                                prepare_generation,
                                ..
                            } => *prepare_generation,
                            _ => None,
                        },
                        ..DiagnosticGraphLifecycleStats::default()
                    },
                );
                continue;
            }
            _ => {}
        }
        let Some(stats) = graphs.get_mut(&key) else {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "conformance lifecycle event precedes graph creation or attachment",
            });
        };
        if stats.dropped_sequence.is_some() {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "conformance lifecycle mutates a dropped graph",
            });
        }
        if stats.poisoned_sequence.is_some()
            && !matches!(event.kind, super::GgmlGraphLifecycleEventKind::Dropped)
        {
            return Err(GgmlCpuGraphError::UnsupportedInputs {
                reason: "conformance lifecycle mutates a poisoned graph",
            });
        }
        match &event.kind {
            super::GgmlGraphLifecycleEventKind::Prepared { prepare_generation } => {
                if stats
                    .prepare_generation
                    .replace(*prepare_generation)
                    .is_some()
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance graph prepared more than once",
                    });
                }
            }
            super::GgmlGraphLifecycleEventKind::InputWrite {
                input_generation, ..
            } => {
                stats.input_generations.insert(*input_generation);
            }
            super::GgmlGraphLifecycleEventKind::ComputeStarted {
                compute_sequence,
                prepare_generation,
                input_generation_consumed,
                capture_executable_generation,
            } => {
                if *compute_sequence == 0
                    || stats.active_compute.is_some()
                    || stats
                        .computes_started
                        .iter()
                        .any(|(observed, ..)| observed == compute_sequence)
                    || stats.capture_generation != *capture_executable_generation
                    || (stats.capture_supported.is_some() && !stats.capture_before_pending)
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance lifecycle contains an invalid compute start",
                    });
                }
                if stats.capture_before_pending {
                    stats.capture_before_computes.insert(*compute_sequence);
                    stats.capture_before_pending = false;
                }
                stats.active_compute = Some(*compute_sequence);
                stats.computes_started.push((
                    *compute_sequence,
                    *prepare_generation,
                    *input_generation_consumed,
                    *capture_executable_generation,
                ));
            }
            super::GgmlGraphLifecycleEventKind::ComputeCompleted {
                compute_sequence,
                output_generation,
            } => {
                if stats.active_compute != Some(*compute_sequence)
                    || (stats.capture_supported.is_some()
                        && !stats.capture_after_computes.contains(compute_sequence))
                    || stats
                        .computes_completed
                        .insert(*compute_sequence, *output_generation)
                        .is_some()
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance lifecycle contains an invalid compute completion",
                    });
                }
                stats.active_compute = None;
            }
            super::GgmlGraphLifecycleEventKind::OutputRead {
                compute_sequence,
                output_generation_consumed,
                ..
            } => stats
                .output_reads
                .push((*compute_sequence, *output_generation_consumed)),
            super::GgmlGraphLifecycleEventKind::KvWriteCommitted {
                compute_sequence, ..
            } => {
                stats.kv_commits.insert(*compute_sequence);
            }
            super::GgmlGraphLifecycleEventKind::Rebuilt {
                previous_graph_generation,
                reason,
            } => {
                if *previous_graph_generation == event.graph_generation
                    || !graph_generations.contains(previous_graph_generation)
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance graph rebuild is not bound to a prior generation",
                    });
                }
                rebuilds.push((event.graph_generation, *previous_graph_generation, *reason));
            }
            super::GgmlGraphLifecycleEventKind::Poisoned { .. } => {
                stats.poisoned_sequence = Some(event.sequence);
                stats.active_compute = None;
                stats.capture_before_pending = false;
            }
            super::GgmlGraphLifecycleEventKind::Dropped => {
                if stats.active_compute.is_some() || stats.capture_before_pending {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance lifecycle drops an active graph compute",
                    });
                }
                stats.dropped_sequence = Some(event.sequence);
            }
            super::GgmlGraphLifecycleEventKind::CaptureStateObserved {
                phase,
                capture_supported,
                graph_tracked,
                capture_enabled,
                executable_present,
            } => {
                if (*graph_tracked && (!*capture_supported || capture_enabled.is_none()))
                    || (!*graph_tracked && capture_enabled.is_some())
                    || (*executable_present && (!*graph_tracked || *capture_enabled != Some(true)))
                    || stats
                        .capture_supported
                        .is_some_and(|previous| previous != *capture_supported)
                    || (stats.graph_tracked == Some(true) && !*graph_tracked)
                    || (stats.graph_tracked == Some(true)
                        && *graph_tracked
                        && stats.capture_enabled != *capture_enabled)
                    || (stats.capture_executable_present && !*executable_present)
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance lifecycle contains drifting native capture state",
                    });
                }
                match phase {
                    super::GgmlCaptureObservationPhase::BeforeCompute => {
                        if stats.active_compute.is_some() || stats.capture_before_pending {
                            return Err(GgmlCpuGraphError::UnsupportedInputs {
                                reason: "conformance capture-before observation is out of phase",
                            });
                        }
                        stats.capture_before_pending = true;
                        stats.capture_before += 1;
                    }
                    super::GgmlCaptureObservationPhase::AfterCompute => {
                        let Some(compute_sequence) = stats.active_compute else {
                            return Err(GgmlCpuGraphError::UnsupportedInputs {
                                reason: "conformance capture-after observation has no active compute",
                            });
                        };
                        if !stats.capture_before_computes.contains(&compute_sequence)
                            || !stats.capture_after_computes.insert(compute_sequence)
                        {
                            return Err(GgmlCpuGraphError::UnsupportedInputs {
                                reason: "conformance capture-after observation is not paired",
                            });
                        }
                        stats.capture_after += 1;
                    }
                }
                stats.capture_supported = Some(*capture_supported);
                stats.graph_tracked = Some(*graph_tracked);
                stats.capture_enabled = *capture_enabled;
                stats.capture_executable_present = *executable_present;
                stats.last_capture_phase = Some(*phase);
                stats.capture_enabled_observed |=
                    *capture_supported && *graph_tracked && *capture_enabled == Some(true);
            }
            super::GgmlGraphLifecycleEventKind::CaptureExecutableObserved {
                capture_executable_generation,
                ..
            } => {
                if *capture_executable_generation == 0
                    || stats.capture_generation.is_some()
                    || stats.active_compute.is_some()
                    || stats.last_capture_phase
                        != Some(super::GgmlCaptureObservationPhase::BeforeCompute)
                    || stats.capture_supported != Some(true)
                    || stats.graph_tracked != Some(true)
                    || stats.capture_enabled != Some(true)
                    || !stats.capture_executable_present
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance pre-compute capture generation is invalid",
                    });
                }
                stats.capture_generation = Some(*capture_executable_generation);
                stats.capture_executable_observed = true;
            }
            super::GgmlGraphLifecycleEventKind::CaptureExecutableCreated {
                capture_executable_generation,
                ..
            } => {
                if *capture_executable_generation == 0
                    || stats
                        .capture_generation
                        .is_some_and(|previous| *capture_executable_generation <= previous)
                    || stats.active_compute.is_none()
                    || stats.last_capture_phase
                        != Some(super::GgmlCaptureObservationPhase::AfterCompute)
                    || stats.capture_supported != Some(true)
                    || stats.graph_tracked != Some(true)
                    || stats.capture_enabled != Some(true)
                    || !stats.capture_executable_present
                {
                    return Err(GgmlCpuGraphError::UnsupportedInputs {
                        reason: "conformance post-compute capture generation is invalid",
                    });
                }
                stats.capture_generation = Some(*capture_executable_generation);
                stats.capture_executable_observed = true;
            }
            super::GgmlGraphLifecycleEventKind::Created { .. }
            | super::GgmlGraphLifecycleEventKind::ExistingGraphObserved { .. } => unreachable!(),
        }
    }
    Ok(DiagnosticLifecycleSummary { graphs, rebuilds })
}

fn validate_computed_graph(
    stats: &DiagnosticGraphLifecycleStats,
    expected_computes: usize,
    require_kv: bool,
    require_capture_observation: bool,
) -> bool {
    if stats.scheduler_enabled != Some(false)
        || stats.prepare_generation.is_none()
        || stats.computes_started.len() != expected_computes
        || stats.computes_completed.len() != expected_computes
        || stats.output_reads.len() != expected_computes
        || stats.dropped_sequence.is_none()
        || stats.active_compute.is_some()
        || stats.capture_before_pending
    {
        return false;
    }
    let mut consumed_inputs = BTreeSet::new();
    for (compute, prepare, input, _) in &stats.computes_started {
        let Some(input) = input else {
            return false;
        };
        let Some(output) = stats.computes_completed.get(compute) else {
            return false;
        };
        if *prepare != stats.prepare_generation
            || !stats.input_generations.contains(input)
            || !stats.output_reads.contains(&(*compute, *output))
        {
            return false;
        }
        consumed_inputs.insert(*input);
    }
    if consumed_inputs.len() != expected_computes {
        return false;
    }
    if require_kv {
        if stats.kv_commits
            != stats
                .computes_started
                .iter()
                .map(|(compute, ..)| *compute)
                .collect()
        {
            return false;
        }
    } else if !stats.kv_commits.is_empty() {
        return false;
    }
    let capture_observed = stats.capture_before > 0 || stats.capture_after > 0;
    if require_capture_observation && !capture_observed {
        return false;
    }
    if capture_observed
        && (stats.capture_before != expected_computes || stats.capture_after != expected_computes)
    {
        return false;
    }
    if stats.capture_enabled_observed {
        if !stats.capture_executable_observed {
            return false;
        }
        // A single compute can only create the executable after it returns.
        // Reuse is proven when a later compute starts with that generation.
        if expected_computes >= 2
            && !stats
                .computes_started
                .iter()
                .any(|(_, _, _, capture)| capture.is_some())
        {
            return false;
        }
    }
    true
}

fn validate_layer2_lifecycle(
    lifecycle: &GgmlGraphLifecycleSnapshot,
    provider: ExecutionProvider,
) -> Result<(), GgmlCpuGraphError> {
    let summary = summarize_lifecycle(lifecycle)?;
    let computed = summary
        .graphs
        .values()
        .filter(|stats| !stats.computes_started.is_empty())
        .collect::<Vec<_>>();
    let kv_graphs = computed
        .iter()
        .filter(|stats| !stats.kv_commits.is_empty())
        .count();
    let poisoned = computed
        .iter()
        .filter(|stats| {
            stats
                .poisoned_sequence
                .zip(stats.dropped_sequence)
                .is_some_and(|(poisoned, dropped)| poisoned < dropped)
        })
        .count();
    let topology_rebuilds = summary
        .rebuilds
        .iter()
        .filter(|(_, _, reason)| *reason == GgmlGraphRebuildReason::TopologyChanged)
        .count();
    let require_capture_observation = provider != ExecutionProvider::Cpu;
    let hip_capture_enabled = provider != ExecutionProvider::Hip
        || computed.iter().all(|stats| stats.capture_enabled_observed);
    if computed.len() != 3
        || kv_graphs != 1
        || poisoned != 1
        || topology_rebuilds != 2
        || !hip_capture_enabled
        || computed.iter().any(|stats| {
            !validate_computed_graph(
                stats,
                2,
                !stats.kv_commits.is_empty(),
                require_capture_observation,
            )
        })
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "Layer-2 lifecycle did not prove reuse, refresh, rebuild, KV, poison, and drop",
        });
    }
    Ok(())
}

fn validate_quadrant_lifecycle(trace: &DiagnosticQuadrantTrace) -> Result<(), GgmlCpuGraphError> {
    let summary = summarize_lifecycle(&trace.graph_lifecycle)?;
    let computed = summary
        .graphs
        .values()
        .filter(|stats| !stats.computes_started.is_empty())
        .collect::<Vec<_>>();
    let valid = match trace.graph_mode {
        DiagnosticDecoderGraphMode::FreshRebuild => {
            computed.len() == trace.steps.len()
                && computed
                    .iter()
                    .all(|stats| validate_computed_graph(stats, 1, false, false))
                && summary
                    .rebuilds
                    .iter()
                    .filter(|(_, _, reason)| *reason == GgmlGraphRebuildReason::FreshStep)
                    .count()
                    == trace.steps.len().saturating_sub(1)
        }
        DiagnosticDecoderGraphMode::ReusableGraph => {
            computed.len() == 1
                && validate_computed_graph(computed[0], trace.steps.len(), false, false)
                && summary.rebuilds.is_empty()
        }
    };
    if !valid {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "four-quadrant lifecycle did not prove the declared fresh or reusable mode",
        });
    }
    Ok(())
}

pub fn classify_four_quadrant_first_divergence(
    input: DiagnosticFourQuadrantClassificationInput<'_>,
) -> DecodeFirstDivergenceClass {
    let a = input.case_a;
    let b = input.case_b;
    if let Some(cpu) = input.cpu_reference {
        if let (Some(c), Some(d)) = (input.case_c, input.case_d)
            && a == b
            && b == c
            && c == d
            && a != cpu
        {
            return DecodeFirstDivergenceClass::EncoderCrossKvAllQuadrants;
        }
        if first_mismatch(a, cpu) == Some(0) {
            return DecodeFirstDivergenceClass::EncoderCrossKvOrKernel;
        }
        let a_ok = a == cpu;
        let b_ok = b == cpu;
        let c_ok = input.case_c.map(|tokens| tokens == cpu);
        let d_ok = input.case_d.map(|tokens| tokens == cpu);
        if a_ok && !b_ok {
            return DecodeFirstDivergenceClass::ReusableKvOrOutputRefresh;
        }
        if a_ok && b_ok && c_ok == Some(false) && d_ok == Some(false) {
            return DecodeFirstDivergenceClass::SelectorOrCompactOutput;
        }
        if a_ok && b_ok && c_ok == Some(true) && d_ok == Some(false) {
            return DecodeFirstDivergenceClass::PersistentCompactInteraction;
        }
        if a_ok && b_ok && c_ok.unwrap_or(true) && d_ok.unwrap_or(true) {
            return DecodeFirstDivergenceClass::NoneObserved;
        }
        return DecodeFirstDivergenceClass::InsufficientEvidence;
    }

    if a != b {
        return DecodeFirstDivergenceClass::ReusableKvOrOutputRefresh;
    }
    if let (Some(c), Some(d)) = (input.case_c, input.case_d) {
        if a != c && a != d {
            return DecodeFirstDivergenceClass::SelectorOrCompactOutput;
        }
        if a == c && a != d {
            return DecodeFirstDivergenceClass::PersistentCompactInteraction;
        }
        if a == c && a == d {
            return DecodeFirstDivergenceClass::NoneObserved;
        }
        return DecodeFirstDivergenceClass::InsufficientEvidence;
    }
    DecodeFirstDivergenceClass::NoneObserved
}

/// CPU-only encoder/decoder split record. Other lanes exist as typed variants
/// but are not executed in this batch.
pub fn synthetic_cpu_encoder_decoder_split_record(
    encoder_row: &[f32],
    decoder_logits_steps: &[&[f32]],
) -> Result<EncoderDecoderSplitProbeRecord, GgmlCpuGraphError> {
    if decoder_logits_steps.is_empty()
        || decoder_logits_steps.len() > SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS
    {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "synthetic encoder/decoder split step count is empty or unbounded",
        });
    }
    let mut host_token_ids = Vec::with_capacity(decoder_logits_steps.len());
    let mut step_logits_hashes = Vec::with_capacity(decoder_logits_steps.len());
    for row in decoder_logits_steps {
        let token =
            diagnostic_host_first_max_token(row).ok_or(GgmlCpuGraphError::UnsupportedInputs {
                reason: "synthetic encoder/decoder split host first-max found no finite logit",
            })?;
        host_token_ids.push(token);
        step_logits_hashes.push(diagnostic_logits_sha256(row));
    }
    Ok(EncoderDecoderSplitProbeRecord {
        lane: EncoderDecoderSplitLane::CpuEncoderCpuDecoder,
        encoder_row_shape: vec![encoder_row.len() as u64],
        encoder_checksum: Some(diagnostic_logits_sha256(encoder_row)),
        layer_tap_tolerance: None,
        cross_kv_checksum: Some(diagnostic_logits_sha256(encoder_row)),
        step_logits_hashes,
        host_token_ids: host_token_ids.clone(),
        device_token_ids: host_token_ids,
        reusable_row_indices: vec![0],
        reusable_positions: vec![0],
        mask_hashes: Vec::new(),
        graph_rebuilt: true,
    })
}

fn run_fresh_complete_logits(
    runner: &mut GgmlCpuGraphRunner,
    steps: &[&[f32]],
) -> Result<DiagnosticQuadrantTrace, GgmlCpuGraphError> {
    let vocab = steps[0].len();
    let mut tokens = Vec::with_capacity(steps.len());
    let mut records = Vec::with_capacity(steps.len());
    for (step, row) in steps.iter().enumerate() {
        let mut graph = runner.start_graph();
        let logits = graph.new_tensor_2d_f32(vocab, 1, "diagnostic_quadrant_logits")?;
        graph.set_input(logits)?;
        graph.set_output(logits)?;
        graph.set_f32_slice(logits, row, "diagnostic_quadrant_logits")?;
        let values = graph.compute_output_f32(logits, vocab)?;
        let token = diagnostic_host_first_max_token(&values).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "fresh complete-logits host first-max found no finite logit",
            },
        )?;
        tokens.push(token);
        records.push(quadrant_step_record(step, token, Some(&values), true)?);
    }
    Ok(DiagnosticQuadrantTrace {
        graph_mode: DiagnosticDecoderGraphMode::FreshRebuild,
        selection: DiagnosticDecodeSelection::CompleteLogitsHostFirstMax,
        tokens,
        steps: records,
        graph_lifecycle: GgmlGraphLifecycleSnapshot::default(),
    })
}

fn run_reusable_complete_logits(
    runner: &mut GgmlCpuGraphRunner,
    steps: &[&[f32]],
) -> Result<DiagnosticQuadrantTrace, GgmlCpuGraphError> {
    let vocab = steps[0].len();
    let mut session = runner.start_persistent_graph_session(1024 * 1024)?;
    let logits = {
        let graph = session.builder();
        let logits = graph.new_tensor_2d_f32(vocab, 1, "diagnostic_quadrant_logits")?;
        graph.set_input(logits)?;
        graph.set_output(logits)?;
        graph.prepare_outputs_for_upload(&[logits])?;
        logits
    };
    let mut tokens = Vec::with_capacity(steps.len());
    let mut records = Vec::with_capacity(steps.len());
    for (step, row) in steps.iter().enumerate() {
        session
            .builder()
            .set_f32_slice(logits, row, "diagnostic_quadrant_logits")?;
        let values = session.builder().compute_output_f32(logits, vocab)?;
        let token = diagnostic_host_first_max_token(&values).ok_or(
            GgmlCpuGraphError::UnsupportedInputs {
                reason: "reusable complete-logits host first-max found no finite logit",
            },
        )?;
        tokens.push(token);
        records.push(quadrant_step_record(step, token, Some(&values), false)?);
    }
    Ok(DiagnosticQuadrantTrace {
        graph_mode: DiagnosticDecoderGraphMode::ReusableGraph,
        selection: DiagnosticDecodeSelection::CompleteLogitsHostFirstMax,
        tokens,
        steps: records,
        graph_lifecycle: GgmlGraphLifecycleSnapshot::default(),
    })
}

fn run_fresh_native_argmax(
    runner: &mut GgmlCpuGraphRunner,
    steps: &[&[f32]],
) -> Result<DiagnosticQuadrantTrace, GgmlCpuGraphError> {
    let vocab = steps[0].len();
    let mut tokens = Vec::with_capacity(steps.len());
    let mut records = Vec::with_capacity(steps.len());
    for (step, row) in steps.iter().enumerate() {
        let mut graph = runner.start_graph();
        let logits = graph.new_tensor_2d_f32(vocab, 1, "diagnostic_quadrant_logits")?;
        graph.set_input(logits)?;
        let token = graph.top1_argmax_first_max(logits)?;
        graph.set_output(token)?;
        graph.set_f32_slice(logits, row, "diagnostic_quadrant_logits")?;
        let selected = graph.compute_output_i32(token, 1)?;
        tokens.push(selected[0]);
        records.push(quadrant_step_record(step, selected[0], None, true)?);
    }
    Ok(DiagnosticQuadrantTrace {
        graph_mode: DiagnosticDecoderGraphMode::FreshRebuild,
        selection: DiagnosticDecodeSelection::NativeArgmaxFirst,
        tokens,
        steps: records,
        graph_lifecycle: GgmlGraphLifecycleSnapshot::default(),
    })
}

fn run_reusable_native_argmax(
    runner: &mut GgmlCpuGraphRunner,
    steps: &[&[f32]],
) -> Result<DiagnosticQuadrantTrace, GgmlCpuGraphError> {
    let vocab = steps[0].len();
    let mut session = start_reusable_native_session(runner, vocab)?;
    let mut tokens = Vec::with_capacity(steps.len());
    let mut records = Vec::with_capacity(steps.len());
    for (step, row) in steps.iter().enumerate() {
        let selected = execute_reusable_native_step(&mut session, row)?;
        tokens.push(selected);
        records.push(quadrant_step_record(step, selected, None, false)?);
    }
    Ok(DiagnosticQuadrantTrace {
        graph_mode: DiagnosticDecoderGraphMode::ReusableGraph,
        selection: DiagnosticDecodeSelection::NativeArgmaxFirst,
        tokens,
        steps: records,
        graph_lifecycle: GgmlGraphLifecycleSnapshot::default(),
    })
}

fn start_reusable_native_session(
    runner: &mut GgmlCpuGraphRunner,
    vocab: usize,
) -> Result<ReusableNativeArgmaxSession, GgmlCpuGraphError> {
    let mut session = runner.start_persistent_graph_session(1024 * 1024)?;
    let (logits, token) = {
        let graph = session.builder();
        let logits = graph.new_tensor_2d_f32(vocab, 1, "diagnostic_quadrant_logits")?;
        graph.set_input(logits)?;
        let token = graph.top1_argmax_first_max(logits)?;
        graph.set_output(token)?;
        graph.prepare_outputs_for_upload(&[token])?;
        (logits, token)
    };
    Ok(ReusableNativeArgmaxSession {
        session,
        logits,
        token,
        vocab,
    })
}

struct ReusableNativeArgmaxSession {
    session: GgmlPersistentGraphSession,
    logits: GgmlCpuTensor<'static>,
    token: GgmlCpuTensor<'static>,
    vocab: usize,
}

fn execute_reusable_native_step(
    session: &mut ReusableNativeArgmaxSession,
    row: &[f32],
) -> Result<i32, GgmlCpuGraphError> {
    if row.len() != session.vocab {
        return Err(GgmlCpuGraphError::UnsupportedInputs {
            reason: "reusable native argmax row width mismatch",
        });
    }
    session
        .session
        .builder()
        .set_f32_slice(session.logits, row, "diagnostic_quadrant_logits")?;
    let selected = session
        .session
        .builder()
        .compute_output_i32(session.token, 1)?;
    Ok(selected[0])
}

fn quadrant_step_record(
    step: usize,
    token: i32,
    logits: Option<&[f32]>,
    graph_rebuilt: bool,
) -> Result<ShortAudioReceiptDecodeStep, GgmlCpuGraphError> {
    let step = u32::try_from(step).map_err(|_| GgmlCpuGraphError::UnsupportedInputs {
        reason: "diagnostic step index exceeds u32",
    })?;
    Ok(ShortAudioReceiptDecodeStep {
        step,
        token_id: Some(token),
        logits_sha256: logits.map(diagnostic_logits_sha256),
        top2_margin: logits
            .and_then(diagnostic_top2)
            .and_then(|top2| top2.margin),
        graph_rebuilt,
    })
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

#[cfg(test)]
fn models_src_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::short_audio_receipt::{
        SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE, SHORT_AUDIO_RECEIPT_SCHEMA, ShortAudioReceipt,
        ShortAudioReceiptAudio, ShortAudioReceiptMetrics, ShortAudioReceiptPack,
        ShortAudioReceiptRun, ShortAudioReceiptTranscript,
    };
    use std::collections::BTreeMap;

    #[test]
    fn diagnostic_dual_output_tie_fixture_and_scalar_refresh() {
        let results = run_diagnostic_dual_output_conformance(&[
            &DIAGNOSTIC_FIRST_MAX_TIE_LOGITS,
            &DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS,
        ])
        .expect("diagnostic dual-output should execute");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].device_token, DIAGNOSTIC_FIRST_MAX_TIE_TOKEN);
        assert_eq!(
            results[0].host_first_max_token,
            DIAGNOSTIC_FIRST_MAX_TIE_TOKEN
        );
        assert!(results[0].tokens_match);
        assert_eq!(results[0].top2.first_index, DIAGNOSTIC_FIRST_MAX_TIE_TOKEN);
        assert_eq!(results[0].top2.first_value, 5.0);
        assert_eq!(results[0].top2.second_index, Some(3));
        assert_eq!(results[0].top2.margin, Some(0.0));
        assert_eq!(results[1].device_token, DIAGNOSTIC_FIRST_MAX_REFRESH_TOKEN);
        assert_eq!(
            results[1].host_first_max_token,
            DIAGNOSTIC_FIRST_MAX_REFRESH_TOKEN
        );
        assert_ne!(results[0].device_token, results[1].device_token);
        assert!(!results[0].authorizes_production_compact());
        assert!(!results[1].authorizes_production_compact());
    }

    #[test]
    fn diagnostic_dual_output_never_authorizes_production_compact() {
        let result = DiagnosticDualOutputConformanceResult {
            logits: DIAGNOSTIC_FIRST_MAX_TIE_LOGITS.to_vec(),
            device_token: DIAGNOSTIC_FIRST_MAX_TIE_TOKEN,
            host_first_max_token: DIAGNOSTIC_FIRST_MAX_TIE_TOKEN,
            tokens_match: true,
            top2: diagnostic_top2(&DIAGNOSTIC_FIRST_MAX_TIE_LOGITS).expect("top2"),
        };
        assert!(!result.authorizes_production_compact());
    }

    #[test]
    fn diagnostic_four_quadrant_cpu_synthetic_agrees() {
        let steps: [&[f32]; 3] = [
            &DIAGNOSTIC_FIRST_MAX_TIE_LOGITS,
            &DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS,
            &[1.0, 4.0, 3.0, 0.0],
        ];
        let report = run_diagnostic_four_quadrant_cpu_probe(
            &steps,
            DiagnosticFamilyCompactPolicy::NativeArgmaxFirstEligible,
        )
        .expect("four-quadrant CPU probe");
        assert_eq!(report.case_a.tokens, vec![2, 0, 1]);
        assert_eq!(report.case_b.tokens, report.case_a.tokens);
        assert_eq!(
            report.case_c.as_ref().map(|trace| trace.tokens.as_slice()),
            Some(report.case_a.tokens.as_slice())
        );
        assert_eq!(
            report.case_d.as_ref().map(|trace| trace.tokens.as_slice()),
            Some(report.case_a.tokens.as_slice())
        );
        assert!(report.case_a.steps.iter().all(|step| step.graph_rebuilt));
        assert!(report.case_b.steps.iter().all(|step| !step.graph_rebuilt));
        assert_eq!(
            report.classification,
            DecodeFirstDivergenceClass::NoneObserved
        );
        assert!(report.graph_lifecycle.events.iter().any(|event| matches!(
            event.kind,
            crate::GgmlGraphLifecycleEventKind::Rebuilt {
                reason: GgmlGraphRebuildReason::FreshStep,
                ..
            }
        )));
    }

    #[test]
    fn exact_route_layer1_binds_real_cpu_provider_and_semantic_cases() {
        let report = run_diagnostic_layer1_exact_route_probe(ResolvedExecutionRoute::cpu())
            .expect("Layer-1 exact CPU probe");
        assert_eq!(report.provider, ExecutionProvider::Cpu);
        assert_eq!(report.stable_device_id, "CPU");
        assert_eq!(report.repeated_refresh_tokens, vec![2, 0]);
        assert!(report.unsupported_type_rejected);
        assert!(report.unsupported_layout_rejected);
        assert!(report.unsupported_rank_rejected);
        assert!(
            report
                .cases
                .iter()
                .all(|case| case.actual_tokens == case.expected_tokens)
        );
        assert!(
            report.graph_lifecycle.events.iter().all(|event| {
                event.provider == "cpu" && event.device == report.stable_device_id
            })
        );
    }

    #[test]
    fn exact_route_layer2_records_refresh_kv_rebuild_poison_and_drop() {
        let report = run_diagnostic_layer2_exact_route_probe(ResolvedExecutionRoute::cpu())
            .expect("Layer-2 exact CPU probe");
        assert_eq!(
            report.full_output_refreshes,
            vec![
                DIAGNOSTIC_FIRST_MAX_TIE_LOGITS.to_vec(),
                DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS.to_vec()
            ]
        );
        assert_eq!(report.scalar_output_refreshes, vec![2, 0]);
        assert_eq!(report.kv_state_refreshes.len(), 2);
        assert!(report.poisoned_reexecution_rejected);
        for expected in [
            "created",
            "prepared",
            "input_write",
            "compute_started",
            "compute_completed",
            "output_read",
            "kv_write_committed",
            "rebuilt",
            "poisoned",
            "dropped",
        ] {
            assert!(
                report
                    .graph_lifecycle
                    .events
                    .iter()
                    .any(|event| lifecycle_event_label(&event.kind) == expected),
                "missing {expected}"
            );
        }
        assert!(report.graph_lifecycle.events.iter().all(|event| !matches!(
            event.kind,
            crate::GgmlGraphLifecycleEventKind::CaptureStateObserved { .. }
                | crate::GgmlGraphLifecycleEventKind::CaptureExecutableObserved { .. }
                | crate::GgmlGraphLifecycleEventKind::CaptureExecutableCreated { .. }
        )));
        assert!(
            validate_layer2_lifecycle(&report.graph_lifecycle, ExecutionProvider::Hip).is_err(),
            "a GPU route cannot pass Layer-2 without native capture observations"
        );
    }

    #[test]
    fn shared_decode_conformance_suite_is_exact_route_and_production_shape() {
        let report = run_diagnostic_decode_conformance_suite(ResolvedExecutionRoute::cpu())
            .expect("shared exact-route CPU conformance suite");
        assert_eq!(report.schema, GPU_DECODE_CONFORMANCE_SCHEMA);
        assert_eq!(report.result, "pass");
        assert_eq!(report.provider, ExecutionProvider::Cpu);
        assert_eq!(report.stable_device_id, "CPU");
        assert_eq!(
            report.production_shape_four_quadrant.vocab_size,
            GPU_DECODE_CONFORMANCE_PRODUCTION_VOCAB
        );
        assert_eq!(report.production_shape_four_quadrant.step_count, 4);
        assert_eq!(
            report.production_shape_four_quadrant.classification,
            DecodeFirstDivergenceClass::NoneObserved
        );
        report.validate().expect("generated suite revalidates");
        let json = serde_json::to_string(&report).expect("serialize suite");
        let decoded: DiagnosticDecodeConformanceSuite =
            serde_json::from_str(&json).expect("strict suite round-trip");
        decoded.validate().expect("round-tripped suite revalidates");

        let mut unknown = serde_json::to_value(&report).expect("suite value");
        unknown
            .as_object_mut()
            .expect("suite object")
            .insert("activation_mode".to_string(), serde_json::json!("auto"));
        assert!(serde_json::from_value::<DiagnosticDecodeConformanceSuite>(unknown).is_err());

        let mut tampered = report;
        tampered.layer1.cases[0].actual_tokens[0] = 2;
        assert!(tampered.validate().is_err());
    }

    #[cfg(feature = "hip")]
    #[test]
    fn shared_decode_conformance_suite_passes_on_live_hip_route() {
        let route = live_hip_route();
        let report = run_diagnostic_decode_conformance_suite(route.clone())
            .expect("shared exact-route HIP conformance suite");
        assert_eq!(report.schema, GPU_DECODE_CONFORMANCE_SCHEMA);
        assert_eq!(report.result, "pass");
        assert_eq!(report.provider, ExecutionProvider::Hip);
        assert_eq!(report.stable_device_id, route.stable_id);
        assert_eq!(
            report.production_shape_four_quadrant.classification,
            DecodeFirstDivergenceClass::NoneObserved
        );
        report.validate().expect("generated HIP suite revalidates");
        validate_layer2_lifecycle(&report.layer2.graph_lifecycle, ExecutionProvider::Hip)
            .expect("HIP Layer-2 must include native capture observations");
        assert!(
            report.layer2.graph_lifecycle.events.iter().any(|event| {
                matches!(
                    event.kind,
                    crate::GgmlGraphLifecycleEventKind::CaptureStateObserved { .. }
                        | crate::GgmlGraphLifecycleEventKind::CaptureExecutableObserved { .. }
                        | crate::GgmlGraphLifecycleEventKind::CaptureExecutableCreated { .. }
                )
            }),
            "HIP Layer-2 must record native capture observations"
        );
    }

    #[cfg(feature = "hip")]
    fn live_hip_route() -> ResolvedExecutionRoute {
        crate::ggml_available_devices()
            .iter()
            .enumerate()
            .find_map(|(ordinal, device)| {
                let row = crate::device::execution_route::enumerated_from_ggml_device(
                    ordinal, device,
                );
                matches!(
                    row.ggml_kind,
                    crate::ggml_runtime::GgmlBackendKind::Gpu
                        | crate::ggml_runtime::GgmlBackendKind::IntegratedGpu
                )
                .then_some(row)
                .filter(|row| row.provider == ExecutionProvider::Hip)
                .map(|row| row.to_resolved_route())
            })
            .expect(
                "HIP feature build must enumerate a live HIP GPU; on Windows use --features hip,legacy-windows-static-sidecar so the probe is compiled into the test binary",
            )
    }

    fn lifecycle_event_label(kind: &crate::GgmlGraphLifecycleEventKind) -> &'static str {
        match kind {
            crate::GgmlGraphLifecycleEventKind::Created { .. } => "created",
            crate::GgmlGraphLifecycleEventKind::ExistingGraphObserved { .. } => {
                "existing_graph_observed"
            }
            crate::GgmlGraphLifecycleEventKind::Prepared { .. } => "prepared",
            crate::GgmlGraphLifecycleEventKind::InputWrite { .. } => "input_write",
            crate::GgmlGraphLifecycleEventKind::ComputeStarted { .. } => "compute_started",
            crate::GgmlGraphLifecycleEventKind::ComputeCompleted { .. } => "compute_completed",
            crate::GgmlGraphLifecycleEventKind::OutputRead { .. } => "output_read",
            crate::GgmlGraphLifecycleEventKind::KvWriteCommitted { .. } => "kv_write_committed",
            crate::GgmlGraphLifecycleEventKind::Rebuilt { .. } => "rebuilt",
            crate::GgmlGraphLifecycleEventKind::Poisoned { .. } => "poisoned",
            crate::GgmlGraphLifecycleEventKind::Dropped => "dropped",
            crate::GgmlGraphLifecycleEventKind::CaptureStateObserved { .. } => {
                "capture_state_observed"
            }
            crate::GgmlGraphLifecycleEventKind::CaptureExecutableObserved { .. } => {
                "capture_executable_observed"
            }
            crate::GgmlGraphLifecycleEventKind::CaptureExecutableCreated { .. } => {
                "capture_executable_created"
            }
        }
    }

    #[test]
    fn diagnostic_xasr_and_mimo_skip_native_compact_quadrants() {
        let steps: [&[f32]; 1] = [&DIAGNOSTIC_FIRST_MAX_TIE_LOGITS];
        for policy in [
            DiagnosticFamilyCompactPolicy::LastMaxHostOracleOnly,
            DiagnosticFamilyCompactPolicy::FirstMaxScoreOracleOnly,
            DiagnosticFamilyCompactPolicy::FullFrameLogitsOnly,
        ] {
            let report = run_diagnostic_four_quadrant_cpu_probe(&steps, policy)
                .expect("A/B-only four-quadrant probe");
            assert!(report.case_c.is_none());
            assert!(report.case_d.is_none());
            assert_eq!(
                report.classification,
                DecodeFirstDivergenceClass::NoneObserved
            );
        }
    }

    #[test]
    fn classify_four_quadrant_contract_cases() {
        let cpu = [1, 2, 3];
        let good = [1, 2, 3];
        let bad_reuse = [1, 9, 3];
        let bad_selector = [4, 5, 6];
        let bad_d = [1, 2, 8];
        let bad_all = [7, 7, 7];

        assert_eq!(
            classify_four_quadrant_first_divergence(DiagnosticFourQuadrantClassificationInput {
                case_a: &good,
                case_b: &bad_reuse,
                case_c: Some(&good),
                case_d: Some(&good),
                cpu_reference: Some(&cpu),
            }),
            DecodeFirstDivergenceClass::ReusableKvOrOutputRefresh
        );
        assert_eq!(
            classify_four_quadrant_first_divergence(DiagnosticFourQuadrantClassificationInput {
                case_a: &good,
                case_b: &good,
                case_c: Some(&bad_selector),
                case_d: Some(&bad_selector),
                cpu_reference: Some(&cpu),
            }),
            DecodeFirstDivergenceClass::SelectorOrCompactOutput
        );
        assert_eq!(
            classify_four_quadrant_first_divergence(DiagnosticFourQuadrantClassificationInput {
                case_a: &bad_all,
                case_b: &good,
                case_c: Some(&good),
                case_d: Some(&good),
                cpu_reference: Some(&cpu),
            }),
            DecodeFirstDivergenceClass::EncoderCrossKvOrKernel
        );
        assert_eq!(
            classify_four_quadrant_first_divergence(DiagnosticFourQuadrantClassificationInput {
                case_a: &good,
                case_b: &good,
                case_c: Some(&good),
                case_d: Some(&bad_d),
                cpu_reference: Some(&cpu),
            }),
            DecodeFirstDivergenceClass::PersistentCompactInteraction
        );
        assert_eq!(
            classify_four_quadrant_first_divergence(DiagnosticFourQuadrantClassificationInput {
                case_a: &bad_all,
                case_b: &bad_all,
                case_c: Some(&bad_all),
                case_d: Some(&bad_all),
                cpu_reference: Some(&cpu),
            }),
            DecodeFirstDivergenceClass::EncoderCrossKvAllQuadrants
        );
    }

    const SAME: EncoderKernelStageChecksumPair<'static> = EncoderKernelStageChecksumPair {
        cpu: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        accel: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    };
    const DIFF: EncoderKernelStageChecksumPair<'static> = EncoderKernelStageChecksumPair {
        cpu: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        accel: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    };

    fn kernel_stage_layer(
        layer_index: usize,
        intra: Option<(
            EncoderKernelStageChecksumPair<'static>,
            EncoderKernelStageChecksumPair<'static>,
            EncoderKernelStageChecksumPair<'static>,
            EncoderKernelStageChecksumPair<'static>,
        )>,
        block_out: EncoderKernelStageChecksumPair<'static>,
    ) -> EncoderKernelStageLayerChecksums<'static> {
        let (ffn1_out, attn_out, conv_out, ffn2_out) = match intra {
            Some((ffn1, attn, conv, ffn2)) => (Some(ffn1), Some(attn), Some(conv), Some(ffn2)),
            None => (None, None, None, None),
        };
        EncoderKernelStageLayerChecksums {
            layer_index,
            ffn1_out,
            attn_out,
            conv_out,
            ffn2_out,
            block_out: Some(block_out),
        }
    }

    fn stem_checksums(
        first_difference: Option<&'static str>,
    ) -> EncoderKernelStageStemChecksums<'static> {
        let pair = |name| {
            if first_difference == Some(name) {
                Some(DIFF)
            } else {
                Some(SAME)
            }
        };
        EncoderKernelStageStemChecksums {
            mel_4d: pair("mel_4d"),
            conv1_raw: pair("conv1_raw"),
            conv1_bias: pair("conv1_bias"),
            conv1_relu: pair("conv1_relu"),
            conv2_raw: pair("conv2_raw"),
            conv2_bias: pair("conv2_bias"),
            conv2_relu: pair("conv2_relu"),
            after_permute: pair("after_permute"),
            after_cont: pair("after_cont"),
            flat_2d: pair("flat_2d"),
            out_matmul: pair("out_matmul"),
            subsample_out: pair("subsample_out"),
        }
    }

    #[test]
    fn classify_encoder_kernel_stage_names_first_bounded_stem_seam() {
        for (tap, class) in [
            ("mel_4d", EncoderKernelStageClass::SubsampleInput),
            ("conv1_raw", EncoderKernelStageClass::SubsampleConvolution),
            ("conv1_bias", EncoderKernelStageClass::SubsampleBias),
            ("conv1_relu", EncoderKernelStageClass::SubsampleRelu),
            ("conv2_raw", EncoderKernelStageClass::SubsampleConvolution),
            ("conv2_bias", EncoderKernelStageClass::SubsampleBias),
            ("conv2_relu", EncoderKernelStageClass::SubsampleRelu),
            ("after_permute", EncoderKernelStageClass::SubsampleLayout),
            ("after_cont", EncoderKernelStageClass::SubsampleLayout),
            ("flat_2d", EncoderKernelStageClass::SubsampleLayout),
            (
                "out_matmul",
                EncoderKernelStageClass::SubsampleOutputProjection,
            ),
            (
                "subsample_out",
                EncoderKernelStageClass::SubsampleOutputProjection,
            ),
        ] {
            let result = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
                stem: Some(stem_checksums(Some(tap))),
                subsample: Some(DIFF),
                layers: &[],
                encoder_output: Some(DIFF),
            });
            assert_eq!(result.class, class, "tap={tap}");
            assert_eq!(result.first_divergent_tap.as_deref(), Some(tap));
            assert_eq!(result.first_divergent_layer, None);
        }
    }

    #[test]
    fn classify_encoder_kernel_stage_rejects_incomplete_stem_evidence() {
        let mut stem = stem_checksums(None);
        stem.conv2_bias = None;
        let result = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: Some(stem),
            subsample: Some(DIFF),
            layers: &[],
            encoder_output: Some(DIFF),
        });
        assert_eq!(result.class, EncoderKernelStageClass::InsufficientEvidence);
        assert_eq!(result.first_divergent_tap.as_deref(), Some("conv2_bias"));
    }

    #[test]
    fn classify_encoder_kernel_stage_contract_cases() {
        let missing_intra = [kernel_stage_layer(0, None, DIFF)];
        let missing = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: None,
            subsample: Some(SAME),
            layers: &missing_intra,
            encoder_output: Some(DIFF),
        });
        assert_eq!(missing.class, EncoderKernelStageClass::InsufficientEvidence);
        assert_eq!(missing.first_divergent_layer, Some(0));

        let subsample_layers = [kernel_stage_layer(0, Some((SAME, DIFF, SAME, SAME)), DIFF)];
        let subsample = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: None,
            subsample: Some(DIFF),
            layers: &subsample_layers,
            encoder_output: Some(DIFF),
        });
        assert_eq!(
            subsample.class,
            EncoderKernelStageClass::InsufficientEvidence
        );
        assert_eq!(
            subsample.first_divergent_tap.as_deref(),
            Some("subsample_out")
        );

        let layer0_same = kernel_stage_layer(0, Some((SAME, SAME, SAME, SAME)), SAME);
        let attn_layers = [
            layer0_same,
            kernel_stage_layer(1, Some((SAME, DIFF, SAME, SAME)), DIFF),
        ];
        let attn = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: None,
            subsample: Some(SAME),
            layers: &attn_layers,
            encoder_output: Some(DIFF),
        });
        assert_eq!(
            attn.class,
            EncoderKernelStageClass::RelativePositionAttention
        );
        assert_eq!(attn.first_divergent_layer, Some(1));
        assert_eq!(attn.first_divergent_tap.as_deref(), Some("attn_out"));

        let conv_layers = [kernel_stage_layer(0, Some((SAME, SAME, DIFF, SAME)), DIFF)];
        let conv = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: None,
            subsample: Some(SAME),
            layers: &conv_layers,
            encoder_output: Some(DIFF),
        });
        assert_eq!(conv.class, EncoderKernelStageClass::DepthwiseConvolution);
        assert_eq!(conv.first_divergent_tap.as_deref(), Some("conv_out"));

        let ffn1_layers = [kernel_stage_layer(0, Some((DIFF, DIFF, SAME, SAME)), DIFF)];
        let ffn1 = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: None,
            subsample: Some(SAME),
            layers: &ffn1_layers,
            encoder_output: Some(DIFF),
        });
        assert_eq!(ffn1.class, EncoderKernelStageClass::InsufficientEvidence);
        assert_eq!(ffn1.first_divergent_tap.as_deref(), Some("ffn1_out"));

        let ffn2_layers = [kernel_stage_layer(0, Some((SAME, SAME, SAME, DIFF)), DIFF)];
        let ffn2 = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: None,
            subsample: Some(SAME),
            layers: &ffn2_layers,
            encoder_output: Some(DIFF),
        });
        assert_eq!(ffn2.class, EncoderKernelStageClass::InsufficientEvidence);
        assert_eq!(ffn2.first_divergent_tap.as_deref(), Some("ffn2_out"));

        let readback_block_layers = [kernel_stage_layer(0, Some((SAME, SAME, SAME, SAME)), DIFF)];
        let readback_block = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: None,
            subsample: Some(SAME),
            layers: &readback_block_layers,
            encoder_output: Some(DIFF),
        });
        assert_eq!(
            readback_block.class,
            EncoderKernelStageClass::EncoderReadback
        );
        assert_eq!(
            readback_block.first_divergent_tap.as_deref(),
            Some("block_out")
        );

        let matching_layers = [kernel_stage_layer(0, Some((SAME, SAME, SAME, SAME)), SAME)];
        let readback_output =
            classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
                stem: None,
                subsample: Some(SAME),
                layers: &matching_layers,
                encoder_output: Some(DIFF),
            });
        assert_eq!(
            readback_output.class,
            EncoderKernelStageClass::EncoderReadback
        );
        assert_eq!(readback_output.first_divergent_layer, None);
        assert_eq!(
            readback_output.first_divergent_tap.as_deref(),
            Some("encoder_output")
        );

        let all_same = classify_encoder_kernel_stage(EncoderKernelStageClassificationInput {
            stem: None,
            subsample: Some(SAME),
            layers: &matching_layers,
            encoder_output: Some(SAME),
        });
        assert_eq!(
            all_same.class,
            EncoderKernelStageClass::InsufficientEvidence
        );
        assert_eq!(all_same.first_divergent_layer, None);
        assert_eq!(all_same.first_divergent_tap, None);
    }

    #[test]
    fn classify_encoder_kernel_stage_serde_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&EncoderKernelStageClass::SubsampleConvolution)
                .expect("serialize class"),
            "\"subsample_convolution\""
        );
        let encoded = serde_json::to_string(&EncoderKernelStageClass::RelativePositionAttention)
            .expect("serialize class");
        assert_eq!(encoded, "\"relative_position_attention\"");
        assert_eq!(
            serde_json::to_string(&EncoderKernelStageClass::DepthwiseConvolution)
                .expect("serialize class"),
            "\"depthwise_convolution\""
        );
        assert_eq!(
            serde_json::to_string(&EncoderKernelStageClass::EncoderReadback)
                .expect("serialize class"),
            "\"encoder_readback\""
        );
        assert_eq!(
            serde_json::to_string(&EncoderKernelStageClass::InsufficientEvidence)
                .expect("serialize class"),
            "\"insufficient_evidence\""
        );
        let hashed = diagnostic_logits_sha256(&[1.0, -0.0, 2.5]);
        assert_eq!(hashed.len(), 64);
        assert_eq!(hashed, diagnostic_logits_sha256(&[1.0, -0.0, 2.5]));
        assert_ne!(hashed, diagnostic_logits_sha256(&[1.0, 0.0, 2.5]));
    }

    #[test]
    fn production_has_no_reverse_gather_or_cuda_vulkan_compact_allowlist() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut hits = Vec::new();
        visit_production_rs(&root, &mut |path, source| {
            if source.contains("top1_argmax_first_max_reversed") {
                hits.push(format!(
                    "{}: top1_argmax_first_max_reversed",
                    path.display()
                ));
            }
            if source.contains("reversed_token_id") {
                hits.push(format!("{}: reversed_token_id", path.display()));
            }
            if source.contains("reverse_index")
                && !source.contains("device_top1_quote_has_no_reverse_index_construction_staging")
            {
                hits.push(format!("{}: reverse_index", path.display()));
            }
        });
        assert!(
            hits.is_empty(),
            "production reverse-gather leftovers remain:\n{}",
            hits.join("\n")
        );
    }

    fn visit_production_rs(dir: &std::path::Path, visit: &mut dyn FnMut(&std::path::Path, &str)) {
        let entries = std::fs::read_dir(dir).expect("read src dir");
        for entry in entries {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("");
                if name == "tests" || name == "third_party" {
                    continue;
                }
                visit_production_rs(&path, visit);
                continue;
            }
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if name == "tests.rs" || name.ends_with("_tests.rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("read rust source");
            let production = strip_cfg_test_modules(&source);
            visit(&path, &production);
        }
    }

    fn strip_cfg_test_modules(source: &str) -> String {
        let mut out = String::with_capacity(source.len());
        let mut skip_depth = 0usize;
        let mut saw_cfg_test = false;
        for line in source.lines() {
            let trimmed = line.trim();
            if skip_depth == 0 && trimmed.starts_with("#[cfg(test)]") {
                saw_cfg_test = true;
                continue;
            }
            if skip_depth == 0 && saw_cfg_test {
                if trimmed.starts_with("#[") {
                    continue;
                }
                if trimmed.starts_with("mod ") || trimmed.starts_with("fn ") {
                    skip_depth = line
                        .chars()
                        .filter(|ch| *ch == '{')
                        .count()
                        .saturating_sub(line.chars().filter(|ch| *ch == '}').count());
                    if skip_depth == 0 && trimmed.ends_with('{') {
                        skip_depth = 1;
                    } else if skip_depth == 0 {
                        saw_cfg_test = false;
                    }
                    continue;
                }
                saw_cfg_test = false;
            }
            if skip_depth > 0 {
                skip_depth = skip_depth
                    .saturating_add(line.chars().filter(|ch| *ch == '{').count())
                    .saturating_sub(line.chars().filter(|ch| *ch == '}').count());
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    #[test]
    fn family_inventory_keeps_xasr_mimo_sensevoice_host_oracles() {
        let root = models_src_root();
        let xasr_head = std::fs::read_to_string(root.join("xasr_zipformer/device_head_graph.rs"))
            .expect("read XASR device head");
        let xasr_greedy = std::fs::read_to_string(root.join("xasr_zipformer/greedy.rs"))
            .expect("read XASR greedy");
        assert!(xasr_head.contains("last-max oracle"));
        assert!(xasr_head.contains("Keep token selection on the host"));
        assert!(xasr_greedy.contains("argmax_uses_last_index_on_exact_ties"));
        assert!(
            xasr_greedy.contains("left.total_cmp(right)"),
            "XASR host oracle must keep last-max max_by ties"
        );
        assert!(
            !xasr_head.contains("top1_argmax_first_max"),
            "XASR must not enter native first-max compact C/D"
        );

        let mimo_rvq =
            std::fs::read_to_string(root.join("mimo_asr/rvq.rs")).expect("read MiMo RVQ");
        let mimo_graph = std::fs::read_to_string(root.join("mimo_asr/audio_tokenizer_graph.rs"))
            .expect("read MiMo tokenizer graph");
        assert!(mimo_rvq.contains("strict first-max"));
        assert!(mimo_rvq.contains("if score > best_score"));
        assert!(mimo_graph.contains("RVQ never uses a device argmax"));
        assert!(
            !mimo_rvq.contains("top1_argmax("),
            "MiMo RVQ must keep the host first-max score oracle"
        );

        let sensevoice = std::fs::read_to_string(root.join("sensevoice/encoder_graph.rs"))
            .expect("read SenseVoice encoder");
        assert!(sensevoice.contains("compute_output_f32(logits, want)"));
        assert!(sensevoice.contains("retains complete per-frame logits"));
        assert!(!sensevoice.contains("FrameTokenIds"));
        assert!(!sensevoice.contains("top1_argmax_first_max"));
    }

    #[test]
    fn diagnostic_host_oracles_keep_family_tie_policies() {
        assert_eq!(
            diagnostic_host_last_max_token(&[3.0, 7.0, 7.0, 2.0]),
            Some(2)
        );
        assert_eq!(
            diagnostic_host_first_max_token(&[3.0, 7.0, 7.0, 2.0]),
            Some(1)
        );
        assert_eq!(
            diagnostic_host_first_max_token(&DIAGNOSTIC_FIRST_MAX_TIE_LOGITS),
            Some(DIAGNOSTIC_FIRST_MAX_TIE_TOKEN)
        );
    }

    #[test]
    fn synthetic_encoder_decoder_split_serializes_into_receipt() {
        let split = synthetic_cpu_encoder_decoder_split_record(
            &[0.25, 0.5, 0.75],
            &[
                &DIAGNOSTIC_FIRST_MAX_TIE_LOGITS,
                &DIAGNOSTIC_FIRST_MAX_REFRESH_LOGITS,
            ],
        )
        .expect("synthetic CPU split");
        assert_eq!(split.lane, EncoderDecoderSplitLane::CpuEncoderCpuDecoder);
        assert_eq!(split.host_token_ids, vec![2, 0]);
        assert_eq!(split.device_token_ids, split.host_token_ids);

        let receipt = sample_receipt_with_diagnostics(ShortAudioReceiptDecodeDiagnostics {
            output_plan: ShortAudioReceiptOutputPlan::FullLogits,
            reuse_mode: ShortAudioReceiptReuseMode::FreshGraph,
            capability_evidence_revision: Some(1),
            steps: vec![ShortAudioReceiptDecodeStep {
                step: 0,
                token_id: Some(2),
                logits_sha256: Some(diagnostic_logits_sha256(&DIAGNOSTIC_FIRST_MAX_TIE_LOGITS)),
                top2_margin: Some(0.0),
                graph_rebuilt: true,
            }],
            first_divergence: Some(DecodeFirstDivergenceClass::NoneObserved),
            encoder_decoder_splits: vec![split],
        });
        let json = receipt.to_pretty_json().expect("serialize receipt");
        let loaded = ShortAudioReceipt::from_json_str(&json).expect("reload receipt");
        let diagnostics = loaded.decode_diagnostics.expect("diagnostics present");
        assert_eq!(
            diagnostics.first_divergence,
            Some(DecodeFirstDivergenceClass::NoneObserved)
        );
        assert_eq!(diagnostics.encoder_decoder_splits.len(), 1);
        assert_eq!(
            diagnostics.encoder_decoder_splits[0].lane,
            EncoderDecoderSplitLane::CpuEncoderCpuDecoder
        );
    }

    #[test]
    fn encoder_decoder_split_lane_names_cover_contract_matrix() {
        let lanes = [
            EncoderDecoderSplitLane::CpuEncoderCpuDecoder,
            EncoderDecoderSplitLane::AccelEncoderCpuDecoder,
            EncoderDecoderSplitLane::CpuEncoderAccelFreshDecoder,
            EncoderDecoderSplitLane::AccelEncoderAccelFreshDecoder,
            EncoderDecoderSplitLane::AccelEncoderAccelReusableDecoder,
        ];
        let encoded = serde_json::to_string(&lanes).expect("serialize lanes");
        assert!(encoded.contains("cpu_encoder_cpu_decoder"));
        assert!(encoded.contains("accel_encoder_cpu_decoder"));
        assert!(encoded.contains("cpu_encoder_accel_fresh_decoder"));
        assert!(encoded.contains("accel_encoder_accel_fresh_decoder"));
        assert!(encoded.contains("accel_encoder_accel_reusable_decoder"));
    }

    #[test]
    fn receipt_output_plan_projects_every_ggml_decode_output_plan() {
        assert_eq!(
            ShortAudioReceiptOutputPlan::from(GgmlDecodeOutputPlan::FullLogits),
            ShortAudioReceiptOutputPlan::FullLogits
        );
        assert_eq!(
            ShortAudioReceiptOutputPlan::from(GgmlDecodeOutputPlan::CompleteScores),
            ShortAudioReceiptOutputPlan::CompleteScores
        );
        assert_eq!(
            ShortAudioReceiptOutputPlan::from(GgmlDecodeOutputPlan::NativeFirstMaxToken),
            ShortAudioReceiptOutputPlan::NativeFirstMaxToken
        );
        assert_eq!(
            ShortAudioReceiptReuseMode::from(GgmlDecodeReuseMode::FreshGraph),
            ShortAudioReceiptReuseMode::FreshGraph
        );
        assert_eq!(
            ShortAudioReceiptReuseMode::from(GgmlDecodeReuseMode::ReusableGraph),
            ShortAudioReceiptReuseMode::ReusableGraph
        );
    }

    fn sample_receipt_with_diagnostics(
        decode_diagnostics: ShortAudioReceiptDecodeDiagnostics,
    ) -> ShortAudioReceipt {
        ShortAudioReceipt::try_new(ShortAudioReceipt {
            schema: SHORT_AUDIO_RECEIPT_SCHEMA.to_string(),
            core_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            pack: ShortAudioReceiptPack {
                model_id: "diagnostic:fp16".to_string(),
                content_sha256: "a".repeat(64),
                size_bytes: 1,
                quant: "fp16".to_string(),
            },
            audio: ShortAudioReceiptAudio {
                path_or_label: "synthetic".to_string(),
                sha256: "b".repeat(64),
                duration_s: None,
            },
            run: ShortAudioReceiptRun {
                backend: "native".to_string(),
                device: "cpu".to_string(),
                os: "darwin".to_string(),
                command: vec!["openasr".to_string(), "probe".to_string()],
                env_allowlist: BTreeMap::new(),
                warmup: "cold".to_string(),
                cache_state: "empty".to_string(),
            },
            metrics: ShortAudioReceiptMetrics {
                wer_or_cer: None,
                rtf_samples: Vec::new(),
                rtf_median: None,
                ttft_s: None,
                peak_rss_bytes: None,
                peak_rss_before_model_bytes: None,
                rss_before_model_bytes: None,
                rss_after_model_bytes: None,
                phys_footprint_before_model_bytes: None,
                phys_footprint_after_model_bytes: None,
                peak_phys_footprint_before_model_bytes: None,
                peak_phys_footprint_bytes: None,
                peak_vram_bytes: None,
                measurement_method: None,
            },
            transcript: ShortAudioReceiptTranscript::from_text(""),
            placement: "cpu".to_string(),
            observed_placement: None,
            graph_lifecycle: None,
            actual_provider: None,
            actual_stable_device_id: None,
            actual_device: None,
            evidence: None,
            execution: None,
            scope: SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE.to_string(),
            notes: Vec::new(),
            decode_diagnostics: Some(decode_diagnostics),
        })
        .expect("diagnostic receipt")
    }
}
