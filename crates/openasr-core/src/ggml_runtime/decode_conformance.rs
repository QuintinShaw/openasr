//! Diagnostic / conformance probes for the GPU decode correctness contract.
//!
//! These helpers exist only to gather Phase 0 evidence. They are not a
//! production output plan, not a lane-capability proof, and must not be wired
//! into a family executor or the shared planner.
//!
//! Dual-output success never authorizes a production compact path. Marking a
//! second output can change ggml allocation and liveness enough to hide a
//! stale-output defect. Production compact authorization requires independent
//! native-only cases C and D compared against a host oracle from a separate
//! full-logits run.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlDecodeOutputPlan, GgmlDecodeReuseMode, GgmlPersistentGraphSession,
};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticDecoderGraphMode {
    FreshRebuild,
    ReusableGraph,
}

/// Selection mode for one four-quadrant cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticQuadrantTrace {
    pub graph_mode: DiagnosticDecoderGraphMode,
    pub selection: DiagnosticDecodeSelection,
    pub tokens: Vec<i32>,
    pub steps: Vec<ShortAudioReceiptDecodeStep>,
}

/// Four-quadrant report produced by two independent CPU runtimes.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagnosticFourQuadrantReport {
    pub case_a: DiagnosticQuadrantTrace,
    pub case_b: DiagnosticQuadrantTrace,
    pub case_c: Option<DiagnosticQuadrantTrace>,
    pub case_d: Option<DiagnosticQuadrantTrace>,
    pub classification: DecodeFirstDivergenceClass,
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

/// Fresh/reuse four-quadrant probe on synthetic logits.
///
/// Uses two independent runtime instances so a fresh rebuild cannot touch the
/// reusable runtime. Cases C/D are omitted when the family must stay on a host
/// oracle (XASR last-max, MiMo RVQ first-max scores, SenseVoice full frames).
pub fn run_diagnostic_four_quadrant_cpu_probe(
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

    let mut fresh_runtime = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())?;
    let mut reusable_runtime = GgmlCpuGraphRunner::new(GgmlCpuGraphConfig::default())?;

    let case_a = run_fresh_complete_logits(&mut fresh_runtime, steps)?;
    let case_b = run_reusable_complete_logits(&mut reusable_runtime, steps)?;
    let (case_c, case_d) = if family_policy.enters_native_compact_quadrants() {
        (
            Some(run_fresh_native_argmax(&mut fresh_runtime, steps)?),
            Some(run_reusable_native_argmax(&mut reusable_runtime, steps)?),
        )
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

    Ok(DiagnosticFourQuadrantReport {
        case_a,
        case_b,
        case_c,
        case_d,
        classification,
    })
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
            evidence: None,
            execution: None,
            scope: SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE.to_string(),
            notes: Vec::new(),
            decode_diagnostics: Some(decode_diagnostics),
        })
        .expect("diagnostic receipt")
    }
}
