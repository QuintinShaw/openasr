//! Machine-readable short-audio audit receipt (`openasr.short-audio-receipt.v0`).
//!
//! Binds the exact core commit, pack bytes, audio fixture, backend/device/OS,
//! command, warmup/cache state, transcript, and optional RTF samples so a
//! later full WER/CER claim can be compared against a frozen short-audio gate.
//!
//! This is data-only evidence for tooling. It is not an execution capability
//! and does not replace [`crate::ModelPackPreflightReceipt`] (pack install
//! sealing).

use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::ggml_runtime::GgmlExecutionPlacementSummary;

pub use crate::ggml_runtime::{
    DecodeFirstDivergenceClass, EncoderDecoderSplitLane, EncoderDecoderSplitProbeRecord,
    SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS, ShortAudioReceiptDecodeDiagnostics,
    ShortAudioReceiptDecodeStep, ShortAudioReceiptOutputPlan, ShortAudioReceiptReuseMode,
};

/// Stable schema id for the short-audio receipt MVP.
pub const SHORT_AUDIO_RECEIPT_SCHEMA: &str = "openasr.short-audio-receipt.v0";

/// Default product scope for the short-audio quality/perf gate.
pub const SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE: &str = "short-audio-gate";

/// How wall time was converted into RTF samples.
pub const SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK: &str = "wall_clock_process_elapsed";

/// Validation failures for a receipt document or its required bindings.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ShortAudioReceiptError {
    #[error("short-audio receipt schema must be {expected}, got {actual:?}")]
    SchemaMismatch {
        expected: &'static str,
        actual: String,
    },
    #[error("short-audio receipt field `{field}` must be non-empty")]
    EmptyField { field: &'static str },
    #[error(
        "short-audio receipt pack.content_sha256 must be 64 lowercase hex chars, got {actual:?}"
    )]
    InvalidContentSha256 { actual: String },
    #[error("short-audio receipt audio.sha256 must be 64 lowercase hex chars, got {actual:?}")]
    InvalidAudioSha256 { actual: String },
    #[error(
        "short-audio receipt transcript.text_sha256 must be 64 lowercase hex chars, got {actual:?}"
    )]
    InvalidTranscriptSha256 { actual: String },
    #[error(
        "short-audio receipt correctness evidence schema must be openasr.short-audio-receipt.evidence.v1, got {actual:?}"
    )]
    EvidenceSchemaMismatch { actual: String },
    #[error("short-audio receipt correctness field `{field}` is invalid: {actual:?}")]
    InvalidEvidenceField { field: &'static str, actual: String },
    #[error(
        "short-audio receipt correctness digest `{field}` must be 64 lowercase hex chars, got {actual:?}"
    )]
    InvalidEvidenceDigest { field: &'static str, actual: String },
    #[error(
        "short-audio receipt correctness evidence class `{evidence_class}` did not pass: {result}"
    )]
    EvidenceNotPassing {
        evidence_class: String,
        result: String,
    },
    #[error("short-audio receipt placement evidence requires observed_placement")]
    PlacementEvidenceMissing,
    #[error("short-audio receipt token-transcript evidence is incomplete")]
    TokenEvidenceIncomplete,
    #[error("short-audio receipt token top-k or margin summary is invalid")]
    InvalidTopKSummary,
    #[error("short-audio receipt core_commit must be a 40-hex git sha, got {actual:?}")]
    InvalidCoreCommit { actual: String },
    #[error("short-audio receipt rtf_median requires non-empty rtf_samples when present")]
    MedianWithoutSamples,
    #[error("short-audio receipt rtf_median {median} does not match samples median {expected}")]
    MedianMismatch { median: String, expected: String },
    #[error("could not hash path {path}: {reason}")]
    HashIo { path: String, reason: String },
    #[error("short-audio receipt decode diagnostics exceed {max} steps, got {actual}")]
    DecodeStepsUnbounded { max: usize, actual: usize },
    #[error(
        "short-audio receipt decode diagnostics exceed {max} encoder/decoder splits, got {actual}"
    )]
    EncoderDecoderSplitsUnbounded { max: usize, actual: usize },
    #[error(
        "short-audio receipt decode diagnostics field `{field}` must be 64 lowercase hex chars, got {actual:?}"
    )]
    InvalidDiagnosticSha256 { field: &'static str, actual: String },
}

/// Top-level short-audio receipt document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortAudioReceipt {
    pub schema: String,
    /// 40-hex git commit of the openasr core that produced the transcript.
    pub core_commit: String,
    pub pack: ShortAudioReceiptPack,
    pub audio: ShortAudioReceiptAudio,
    pub run: ShortAudioReceiptRun,
    pub metrics: ShortAudioReceiptMetrics,
    pub transcript: ShortAudioReceiptTranscript,
    /// Reported weight/compute placement label (requested device in v0 when
    /// runtime placement is not introspected).
    pub placement: String,
    /// Actual graph-node placement observed at compute time. Older receipts
    /// and non-ggml backends may omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_placement: Option<GgmlExecutionPlacementSummary>,
    /// Optional versioned evidence. Its class determines which release gate it
    /// can satisfy; old receipts omit this field and remain readable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<ShortAudioReceiptEvidence>,
    /// Gate scope, typically [`SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE`].
    pub scope: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Optional decode-correctness diagnostics. Absent on older v0 receipts.
    /// Dual-output or four-quadrant agreement recorded here is not production
    /// compact-path authorization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_diagnostics: Option<ShortAudioReceiptDecodeDiagnostics>,
}

/// Pack identity bound into the receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortAudioReceiptPack {
    pub model_id: String,
    /// Lowercase hex sha256 of the exact pack bytes (no `sha256:` prefix).
    pub content_sha256: String,
    pub size_bytes: u64,
    pub quant: String,
}

/// Audio fixture bound into the receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortAudioReceiptAudio {
    pub path_or_label: String,
    /// Lowercase hex sha256 of the exact audio file bytes.
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<f64>,
}

/// Run environment and command binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortAudioReceiptRun {
    /// CLI backend kind (`native` / `mock`).
    pub backend: String,
    /// Device label requested for the run (`cpu` / `metal` / `cuda` / `auto` / ...).
    pub device: String,
    /// Host OS id: `darwin`, `linux`, or `windows`.
    pub os: String,
    /// Effective command argv that produced the receipt (tooling-facing).
    pub command: Vec<String>,
    /// Small allowlisted environment snapshot (never a full env dump).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env_allowlist: BTreeMap<String, String>,
    /// `cold` or `warm` relative to process / model-cache state.
    pub warmup: String,
    /// `empty` or `populated` cache state at the timed runs.
    pub cache_state: String,
}

/// Optional metrics. Empty RTF lists are valid for transcript-only receipts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortAudioReceiptMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wer_or_cer: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rtf_samples: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtf_median: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_bytes: Option<u64>,
    /// Process peak RSS captured immediately before the first model run. The
    /// difference to `peak_rss_bytes` isolates additional high-water created
    /// by model execution from CLI/audio preparation startup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_before_model_bytes: Option<u64>,
    /// Process RSS after audio preparation but before the first model run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rss_before_model_bytes: Option<u64>,
    /// Process RSS after all warmup and measured runs, while resident runtime
    /// caches still reflect the product's warm state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rss_after_model_bytes: Option<u64>,
    /// Darwin process physical footprint before the first model run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phys_footprint_before_model_bytes: Option<u64>,
    /// Darwin process physical footprint after all model runs while warm
    /// runtimes remain resident.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phys_footprint_after_model_bytes: Option<u64>,
    /// Darwin lifetime maximum physical footprint before the first model run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_phys_footprint_before_model_bytes: Option<u64>,
    /// Darwin lifetime maximum physical footprint after all model runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_phys_footprint_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_vram_bytes: Option<u64>,
    /// How RTF was measured. v0 uses wall-clock process elapsed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement_method: Option<String>,
}

/// Transcript payload and content hash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortAudioReceiptTranscript {
    pub text: String,
    /// Lowercase hex sha256 of the UTF-8 transcript bytes.
    pub text_sha256: String,
}

impl ShortAudioReceipt {
    /// Build a receipt and validate required bindings.
    pub fn try_new(mut receipt: ShortAudioReceipt) -> Result<Self, ShortAudioReceiptError> {
        if receipt.metrics.measurement_method.is_none() && !receipt.metrics.rtf_samples.is_empty() {
            receipt.metrics.measurement_method =
                Some(SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK.to_string());
        }
        if receipt.metrics.rtf_median.is_none() && !receipt.metrics.rtf_samples.is_empty() {
            receipt.metrics.rtf_median = median_f64(&receipt.metrics.rtf_samples);
        }
        receipt.validate()?;
        Ok(receipt)
    }

    /// Fail-closed field checks for tooling that loads a receipt from disk.
    pub fn validate(&self) -> Result<(), ShortAudioReceiptError> {
        if self.schema != SHORT_AUDIO_RECEIPT_SCHEMA {
            return Err(ShortAudioReceiptError::SchemaMismatch {
                expected: SHORT_AUDIO_RECEIPT_SCHEMA,
                actual: self.schema.clone(),
            });
        }
        require_non_empty("core_commit", &self.core_commit)?;
        validate_core_commit(&self.core_commit)?;
        require_non_empty("pack.model_id", &self.pack.model_id)?;
        require_non_empty("pack.quant", &self.pack.quant)?;
        validate_sha256_hex("pack.content_sha256", &self.pack.content_sha256)
            .map_err(|actual| ShortAudioReceiptError::InvalidContentSha256 { actual })?;
        require_non_empty("audio.path_or_label", &self.audio.path_or_label)?;
        validate_sha256_hex("audio.sha256", &self.audio.sha256)
            .map_err(|actual| ShortAudioReceiptError::InvalidAudioSha256 { actual })?;
        require_non_empty("run.backend", &self.run.backend)?;
        require_non_empty("run.device", &self.run.device)?;
        require_non_empty("run.os", &self.run.os)?;
        require_non_empty("run.warmup", &self.run.warmup)?;
        require_non_empty("run.cache_state", &self.run.cache_state)?;
        if self.run.command.is_empty() || self.run.command.iter().any(|part| part.trim().is_empty())
        {
            return Err(ShortAudioReceiptError::EmptyField {
                field: "run.command",
            });
        }
        require_non_empty("placement", &self.placement)?;
        require_non_empty("scope", &self.scope)?;
        validate_sha256_hex("transcript.text_sha256", &self.transcript.text_sha256)
            .map_err(|actual| ShortAudioReceiptError::InvalidTranscriptSha256 { actual })?;
        let expected_text_sha = sha256_hex_bytes(self.transcript.text.as_bytes());
        if self.transcript.text_sha256 != expected_text_sha {
            return Err(ShortAudioReceiptError::InvalidTranscriptSha256 {
                actual: self.transcript.text_sha256.clone(),
            });
        }

        if let Some(evidence) = &self.evidence {
            evidence.validate(self.observed_placement.as_ref())?;
        }

        match (self.metrics.rtf_median, self.metrics.rtf_samples.is_empty()) {
            (Some(_), true) => return Err(ShortAudioReceiptError::MedianWithoutSamples),
            (Some(median), false) => {
                let expected = median_f64(&self.metrics.rtf_samples)
                    .ok_or(ShortAudioReceiptError::MedianWithoutSamples)?;
                if !approx_eq(median, expected) {
                    return Err(ShortAudioReceiptError::MedianMismatch {
                        median: format!("{median}"),
                        expected: format!("{expected}"),
                    });
                }
            }
            (None, _) => {}
        }
        if let Some(diagnostics) = &self.decode_diagnostics {
            validate_decode_diagnostics(diagnostics)?;
        }
        Ok(())
    }

    /// Serialize as pretty JSON.
    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parse JSON and validate required fields.
    pub fn from_json_str(raw: &str) -> Result<Self, ShortAudioReceiptLoadError> {
        let receipt: Self = serde_json::from_str(raw)?;
        receipt.validate()?;
        Ok(receipt)
    }
}

/// Load-time errors (serde or validation).
#[derive(Debug, Error)]
pub enum ShortAudioReceiptLoadError {
    #[error(transparent)]
    Serde(#[from] serde_json::Error),
    #[error(transparent)]
    Validate(#[from] ShortAudioReceiptError),
}

impl ShortAudioReceiptTranscript {
    pub fn from_text(text: impl Into<String>) -> Self {
        let text = text.into();
        let text_sha256 = sha256_hex_bytes(text.as_bytes());
        Self { text, text_sha256 }
    }
}

/// optional extension makes old receipts readable while making their absence
/// explicit: a plain short-audio receipt is never GPU token-correctness proof.
pub const SHORT_AUDIO_RECEIPT_EVIDENCE_SCHEMA: &str = "openasr.short-audio-receipt.evidence.v1";

/// Evidence classes are intentionally disjoint. A passing placement receipt
/// cannot be consumed as token correctness, and a packaging receipt cannot
/// authorize runtime placement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortAudioReceiptEvidence {
    pub schema: String,
    pub evidence_class: String,
    pub family: String,
    pub provider: String,
    /// Stable bounded device label or opaque identity; never a local path.
    pub device: String,
    pub placement: String,
    pub result: String,
    pub artifacts: ShortAudioReceiptArtifacts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_plan: Option<ShortAudioOutputPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family_oracle: Option<ShortAudioFamilyOracle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ShortAudioExecutionMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<ShortAudioTraceSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortAudioReceiptArtifacts {
    pub binary: ShortAudioArtifactIdentity,
    pub pack: ShortAudioArtifactIdentity,
    pub fixture: ShortAudioArtifactIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortAudioArtifactIdentity {
    pub label: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortAudioOutputPlan {
    pub kind: String,
    pub logits_or_scores: String,
    pub tie_policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortAudioFamilyOracle {
    pub family: String,
    pub tie_policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortAudioExecutionMode {
    pub mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_rebuild_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortAudioTraceSummary {
    pub token_trace_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_k: Vec<ShortAudioTopKSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top1_top2_margin: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShortAudioTopKSummary {
    pub token_id: u32,
    pub value: f64,
}

impl ShortAudioReceiptEvidence {
    fn validate(
        &self,
        observed_placement: Option<&GgmlExecutionPlacementSummary>,
    ) -> Result<(), ShortAudioReceiptError> {
        if self.schema != SHORT_AUDIO_RECEIPT_EVIDENCE_SCHEMA {
            return Err(ShortAudioReceiptError::EvidenceSchemaMismatch {
                actual: self.schema.clone(),
            });
        }
        for (field, value) in [
            ("correctness.evidence_class", self.evidence_class.as_str()),
            ("correctness.family", self.family.as_str()),
            ("correctness.provider", self.provider.as_str()),
            ("correctness.device", self.device.as_str()),
            ("correctness.placement", self.placement.as_str()),
            ("correctness.result", self.result.as_str()),
        ] {
            require_non_empty(field, value)?;
            if value.len() > 128 || value.contains(['\n', '\r']) {
                return Err(ShortAudioReceiptError::InvalidEvidenceField {
                    field,
                    actual: value.to_string(),
                });
            }
        }
        if self.result != "pass" {
            return Err(ShortAudioReceiptError::EvidenceNotPassing {
                evidence_class: self.evidence_class.clone(),
                result: self.result.clone(),
            });
        }
        self.artifacts.validate()?;
        match self.evidence_class.as_str() {
            "build_packaging" => {}
            "placement_resource" => {
                if observed_placement.is_none() {
                    return Err(ShortAudioReceiptError::PlacementEvidenceMissing);
                }
            }
            "token_transcript" => {
                if self.output_plan.is_none()
                    || self.family_oracle.is_none()
                    || self.execution.is_none()
                    || self.trace.is_none()
                {
                    return Err(ShortAudioReceiptError::TokenEvidenceIncomplete);
                }
                let execution = self.execution.as_ref().expect("checked above");
                if execution.mode != "fresh" && execution.mode != "reuse" {
                    return Err(ShortAudioReceiptError::InvalidEvidenceField {
                        field: "correctness.execution.mode",
                        actual: execution.mode.clone(),
                    });
                }
                let trace = self.trace.as_ref().expect("checked above");
                validate_sha256_hex(
                    "correctness.trace.token_trace_sha256",
                    &trace.token_trace_sha256,
                )
                .map_err(|actual| {
                    ShortAudioReceiptError::InvalidEvidenceDigest {
                        field: "correctness.trace.token_trace_sha256",
                        actual,
                    }
                })?;
                if let Some(logits) = &trace.logits_sha256 {
                    validate_sha256_hex("correctness.trace.logits_sha256", logits).map_err(
                        |actual| ShortAudioReceiptError::InvalidEvidenceDigest {
                            field: "correctness.trace.logits_sha256",
                            actual,
                        },
                    )?;
                }
                if trace.top_k.len() > 32 || trace.top_k.iter().any(|item| !item.value.is_finite())
                {
                    return Err(ShortAudioReceiptError::InvalidTopKSummary);
                }
                if trace
                    .top1_top2_margin
                    .is_some_and(|margin| !margin.is_finite() || margin < 0.0)
                {
                    return Err(ShortAudioReceiptError::InvalidTopKSummary);
                }
            }
            _ => {
                return Err(ShortAudioReceiptError::InvalidEvidenceField {
                    field: "correctness.evidence_class",
                    actual: self.evidence_class.clone(),
                });
            }
        }
        Ok(())
    }
}

impl ShortAudioReceiptArtifacts {
    fn validate(&self) -> Result<(), ShortAudioReceiptError> {
        for (field, identity) in [
            ("correctness.artifacts.binary", &self.binary),
            ("correctness.artifacts.pack", &self.pack),
            ("correctness.artifacts.fixture", &self.fixture),
        ] {
            require_non_empty(field, &identity.label)?;
            if identity.label.len() > 128 || identity.label.contains(['\n', '\r', '/', '\\']) {
                return Err(ShortAudioReceiptError::InvalidEvidenceField {
                    field,
                    actual: identity.label.clone(),
                });
            }
            validate_sha256_hex("correctness.artifacts.sha256", &identity.sha256).map_err(
                |actual| ShortAudioReceiptError::InvalidEvidenceDigest {
                    field: "correctness.artifacts.sha256",
                    actual,
                },
            )?;
        }
        Ok(())
    }
}

/// Lowercase hex sha256 of `bytes`.
pub fn sha256_hex_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_lower(digest)
}

/// Stream-hash a file without loading it entirely into memory.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<(u64, String), ShortAudioReceiptError> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|source| ShortAudioReceiptError::HashIo {
        path: path.display().to_string(),
        reason: source.to_string(),
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0_u8; 1024 * 64];
    let mut total = 0_u64;
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|source| ShortAudioReceiptError::HashIo {
                path: path.display().to_string(),
                reason: source.to_string(),
            })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total = total.saturating_add(n as u64);
    }
    Ok((total, hex_lower(hasher.finalize())))
}

/// Median of f64 samples. Even count uses the mean of the two central values.
pub fn median_f64(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        Some(sorted[mid])
    } else {
        Some((sorted[mid - 1] + sorted[mid]) / 2.0)
    }
}

/// Compact host OS id used in receipts: `darwin` / `linux` / `windows`.
pub fn receipt_os_id() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "darwin"
    }
    #[cfg(target_os = "linux")]
    {
        "linux"
    }
    #[cfg(windows)]
    {
        "windows"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
    {
        "unknown"
    }
}

/// Validate a 40-hex git commit sha (lowercase or uppercase accepted; stored as-is).
pub fn validate_core_commit(value: &str) -> Result<(), ShortAudioReceiptError> {
    let ok = value.len() == 40 && value.bytes().all(|b| b.is_ascii_hexdigit());
    if ok {
        Ok(())
    } else {
        Err(ShortAudioReceiptError::InvalidCoreCommit {
            actual: value.to_string(),
        })
    }
}

/// Resolve core commit from explicit value, then `OPENASR_BUILD_COMMIT`, then
/// `git rev-parse HEAD` in `git_cwd` when provided.
pub fn resolve_core_commit(
    explicit: Option<&str>,
    git_cwd: Option<&Path>,
) -> Result<String, ShortAudioReceiptError> {
    if let Some(value) = explicit.map(str::trim).filter(|v| !v.is_empty()) {
        validate_core_commit(value)?;
        return Ok(value.to_ascii_lowercase());
    }
    if let Ok(value) = std::env::var(crate::ggml_runtime::BUILD_COMMIT_ENV) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            validate_core_commit(trimmed)?;
            return Ok(trimmed.to_ascii_lowercase());
        }
    }
    if let Some(cwd) = git_cwd
        && let Some(sha) = git_rev_parse_head(cwd)
    {
        validate_core_commit(&sha)?;
        return Ok(sha.to_ascii_lowercase());
    }
    Err(ShortAudioReceiptError::InvalidCoreCommit {
        actual: String::new(),
    })
}

fn git_rev_parse_head(cwd: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8(output.stdout).ok()?;
    let sha = sha.trim();
    if sha.len() == 40 {
        Some(sha.to_string())
    } else {
        None
    }
}

fn require_non_empty(field: &'static str, value: &str) -> Result<(), ShortAudioReceiptError> {
    if value.trim().is_empty() {
        Err(ShortAudioReceiptError::EmptyField { field })
    } else {
        Ok(())
    }
}

const MAX_ENCODER_DECODER_SPLITS: usize = 8;

fn validate_decode_diagnostics(
    diagnostics: &ShortAudioReceiptDecodeDiagnostics,
) -> Result<(), ShortAudioReceiptError> {
    if diagnostics.steps.len() > SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS {
        return Err(ShortAudioReceiptError::DecodeStepsUnbounded {
            max: SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS,
            actual: diagnostics.steps.len(),
        });
    }
    if diagnostics.encoder_decoder_splits.len() > MAX_ENCODER_DECODER_SPLITS {
        return Err(ShortAudioReceiptError::EncoderDecoderSplitsUnbounded {
            max: MAX_ENCODER_DECODER_SPLITS,
            actual: diagnostics.encoder_decoder_splits.len(),
        });
    }
    for step in &diagnostics.steps {
        if let Some(hash) = &step.logits_sha256 {
            validate_sha256_hex("decode_diagnostics.steps.logits_sha256", hash).map_err(
                |actual| ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.steps.logits_sha256",
                    actual,
                },
            )?;
        }
    }
    for split in &diagnostics.encoder_decoder_splits {
        if split.step_logits_hashes.len() > SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS {
            return Err(ShortAudioReceiptError::DecodeStepsUnbounded {
                max: SHORT_AUDIO_RECEIPT_MAX_DECODE_STEPS,
                actual: split.step_logits_hashes.len(),
            });
        }
        if let Some(hash) = &split.encoder_checksum {
            validate_sha256_hex("decode_diagnostics.encoder_checksum", hash).map_err(|actual| {
                ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.encoder_checksum",
                    actual,
                }
            })?;
        }
        if let Some(hash) = &split.cross_kv_checksum {
            validate_sha256_hex("decode_diagnostics.cross_kv_checksum", hash).map_err(
                |actual| ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.cross_kv_checksum",
                    actual,
                },
            )?;
        }
        for hash in &split.step_logits_hashes {
            validate_sha256_hex("decode_diagnostics.step_logits_hashes", hash).map_err(
                |actual| ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.step_logits_hashes",
                    actual,
                },
            )?;
        }
        for hash in &split.mask_hashes {
            validate_sha256_hex("decode_diagnostics.mask_hashes", hash).map_err(|actual| {
                ShortAudioReceiptError::InvalidDiagnosticSha256 {
                    field: "decode_diagnostics.mask_hashes",
                    actual,
                }
            })?;
        }
    }
    Ok(())
}

fn validate_sha256_hex(_field: &str, value: &str) -> Result<(), String> {
    let ok = value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if ok { Ok(()) } else { Err(value.to_string()) }
}

fn hex_lower(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn approx_eq(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= scale * 1e-9
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn sample_receipt() -> ShortAudioReceipt {
        let transcript = ShortAudioReceiptTranscript::from_text("hello world");
        ShortAudioReceipt {
            schema: SHORT_AUDIO_RECEIPT_SCHEMA.to_string(),
            core_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            pack: ShortAudioReceiptPack {
                model_id: "funasr-nano:q4_k".to_string(),
                content_sha256: "a".repeat(64),
                size_bytes: 12,
                quant: "q4_k".to_string(),
            },
            audio: ShortAudioReceiptAudio {
                path_or_label: "fixtures/jfk.wav".to_string(),
                sha256: "b".repeat(64),
                duration_s: Some(1.5),
            },
            run: ShortAudioReceiptRun {
                backend: "native".to_string(),
                device: "cpu".to_string(),
                os: "darwin".to_string(),
                command: vec![
                    "openasr".to_string(),
                    "bench-receipt".to_string(),
                    "short-audio".to_string(),
                ],
                env_allowlist: BTreeMap::from([(
                    "OPENASR_HOME".to_string(),
                    "/tmp/isolated".to_string(),
                )]),
                warmup: "cold".to_string(),
                cache_state: "empty".to_string(),
            },
            metrics: ShortAudioReceiptMetrics {
                wer_or_cer: None,
                rtf_samples: vec![0.4, 0.5, 0.6],
                rtf_median: Some(0.5),
                ttft_s: None,
                peak_rss_bytes: Some(1024),
                peak_rss_before_model_bytes: Some(640),
                rss_before_model_bytes: Some(512),
                rss_after_model_bytes: Some(768),
                phys_footprint_before_model_bytes: Some(448),
                phys_footprint_after_model_bytes: Some(704),
                peak_phys_footprint_before_model_bytes: Some(576),
                peak_phys_footprint_bytes: Some(896),
                peak_vram_bytes: None,
                measurement_method: Some(SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK.to_string()),
            },
            transcript,
            placement: "cpu".to_string(),
            observed_placement: Some(GgmlExecutionPlacementSummary {
                direct_graph_computes: 1,
                scheduler_graph_computes: 0,
                observed_nodes_by_backend: BTreeMap::from([("CPU".to_string(), 12)]),
                observed_compute_nodes_by_backend: BTreeMap::from([("CPU".to_string(), 10)]),
                observed_node_output_bytes_by_backend: BTreeMap::from([("CPU".to_string(), 4096)]),
                fallback_node_samples_by_backend: BTreeMap::new(),
            }),
            evidence: None,
            scope: SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE.to_string(),
            notes: vec!["unit-test fixture".to_string()],
            decode_diagnostics: None,
        }
    }

    fn sample_artifacts() -> ShortAudioReceiptArtifacts {
        ShortAudioReceiptArtifacts {
            binary: ShortAudioArtifactIdentity {
                label: "openasr-test-binary".to_string(),
                sha256: "c".repeat(64),
                size_bytes: Some(10),
            },
            pack: ShortAudioArtifactIdentity {
                label: "fixture-pack".to_string(),
                sha256: "d".repeat(64),
                size_bytes: Some(20),
            },
            fixture: ShortAudioArtifactIdentity {
                label: "jfk-short".to_string(),
                sha256: "e".repeat(64),
                size_bytes: Some(30),
            },
        }
    }

    fn sample_token_evidence(mode: &str) -> ShortAudioReceiptEvidence {
        ShortAudioReceiptEvidence {
            schema: SHORT_AUDIO_RECEIPT_EVIDENCE_SCHEMA.to_string(),
            evidence_class: "token_transcript".to_string(),
            family: "qwen".to_string(),
            provider: "cuda".to_string(),
            device: "cuda0".to_string(),
            placement: "full_device".to_string(),
            result: "pass".to_string(),
            artifacts: sample_artifacts(),
            output_plan: Some(ShortAudioOutputPlan {
                kind: "full_logits".to_string(),
                logits_or_scores: "complete".to_string(),
                tie_policy: "first_maximum".to_string(),
            }),
            family_oracle: Some(ShortAudioFamilyOracle {
                family: "qwen".to_string(),
                tie_policy: "first_maximum".to_string(),
            }),
            execution: Some(ShortAudioExecutionMode {
                mode: mode.to_string(),
                graph_rebuild_reason: None,
            }),
            trace: Some(ShortAudioTraceSummary {
                token_trace_sha256: "f".repeat(64),
                logits_sha256: Some("a".repeat(64)),
                top_k: vec![ShortAudioTopKSummary {
                    token_id: 7,
                    value: 1.25,
                }],
                top1_top2_margin: Some(0.5),
            }),
        }
    }

    #[test]
    fn token_evidence_validates_as_a_separate_class() {
        let mut receipt = sample_receipt();
        receipt.evidence = Some(sample_token_evidence("fresh"));
        ShortAudioReceipt::try_new(receipt).expect("complete token evidence should validate");
    }

    #[test]
    fn token_evidence_rejects_missing_trace() {
        let mut receipt = sample_receipt();
        let mut evidence = sample_token_evidence("reuse");
        evidence.trace = None;
        receipt.evidence = Some(evidence);
        assert!(matches!(
            receipt.validate(),
            Err(ShortAudioReceiptError::TokenEvidenceIncomplete)
        ));
    }

    #[test]
    fn placement_evidence_cannot_approve_without_observed_placement() {
        let mut receipt = sample_receipt();
        let mut evidence = sample_token_evidence("fresh");
        evidence.evidence_class = "placement_resource".to_string();
        evidence.output_plan = None;
        evidence.family_oracle = None;
        evidence.execution = None;
        evidence.trace = None;
        receipt.observed_placement = None;
        receipt.evidence = Some(evidence);
        assert!(matches!(
            receipt.validate(),
            Err(ShortAudioReceiptError::PlacementEvidenceMissing)
        ));
    }
    #[test]
    fn roundtrip_json_preserves_receipt() {
        let receipt = ShortAudioReceipt::try_new(sample_receipt()).unwrap();
        let json = receipt.to_pretty_json().unwrap();
        let loaded = ShortAudioReceipt::from_json_str(&json).unwrap();
        assert_eq!(loaded.schema, SHORT_AUDIO_RECEIPT_SCHEMA);
        assert_eq!(loaded.pack.model_id, "funasr-nano:q4_k");
        assert_eq!(loaded.metrics.rtf_median, Some(0.5));
        assert_eq!(loaded.transcript.text, "hello world");
        assert_eq!(
            loaded.transcript.text_sha256,
            sha256_hex_bytes(b"hello world")
        );
    }

    #[test]
    fn missing_required_field_fails_validation() {
        let mut receipt = sample_receipt();
        receipt.core_commit.clear();
        let err = receipt.validate().unwrap_err();
        assert!(matches!(
            err,
            ShortAudioReceiptError::EmptyField {
                field: "core_commit"
            }
        ));
    }

    #[test]
    fn wrong_schema_fails_validation() {
        let mut receipt = sample_receipt();
        receipt.schema = "openasr.model-pack-preflight.v1".to_string();
        let err = receipt.validate().unwrap_err();
        assert!(matches!(err, ShortAudioReceiptError::SchemaMismatch { .. }));
    }

    #[test]
    fn transcript_sha_mismatch_fails() {
        let mut receipt = sample_receipt();
        receipt.transcript.text_sha256 = "c".repeat(64);
        let err = receipt.validate().unwrap_err();
        assert!(matches!(
            err,
            ShortAudioReceiptError::InvalidTranscriptSha256 { .. }
        ));
    }

    #[test]
    fn empty_rtf_samples_are_allowed() {
        let mut receipt = sample_receipt();
        receipt.metrics.rtf_samples.clear();
        receipt.metrics.rtf_median = None;
        ShortAudioReceipt::try_new(receipt).unwrap();
    }

    #[test]
    fn median_without_samples_fails() {
        let mut receipt = sample_receipt();
        receipt.metrics.rtf_samples.clear();
        receipt.metrics.rtf_median = Some(0.5);
        let err = receipt.validate().unwrap_err();
        assert!(matches!(err, ShortAudioReceiptError::MedianWithoutSamples));
    }

    #[test]
    fn median_helper_handles_even_and_odd() {
        assert_eq!(median_f64(&[]), None);
        assert_eq!(median_f64(&[3.0]), Some(3.0));
        assert_eq!(median_f64(&[1.0, 3.0, 2.0]), Some(2.0));
        assert_eq!(median_f64(&[1.0, 2.0, 3.0, 4.0]), Some(2.5));
    }

    #[test]
    fn sha256_file_matches_bytes_helper() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"short-audio-receipt").unwrap();
        tmp.flush().unwrap();
        let (size, hex) = sha256_file(tmp.path()).unwrap();
        assert_eq!(size, b"short-audio-receipt".len() as u64);
        assert_eq!(hex, sha256_hex_bytes(b"short-audio-receipt"));
    }

    #[test]
    fn resolve_core_commit_accepts_explicit_sha() {
        let sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(resolve_core_commit(Some(sha), None).unwrap(), sha);
    }

    #[test]
    fn resolve_core_commit_rejects_short_sha() {
        let err = resolve_core_commit(Some("abc"), None).unwrap_err();
        assert!(matches!(
            err,
            ShortAudioReceiptError::InvalidCoreCommit { .. }
        ));
    }

    #[test]
    fn try_new_fills_median_and_measurement_method() {
        let mut receipt = sample_receipt();
        receipt.metrics.rtf_median = None;
        receipt.metrics.measurement_method = None;
        let built = ShortAudioReceipt::try_new(receipt).unwrap();
        assert_eq!(built.metrics.rtf_median, Some(0.5));
        assert_eq!(
            built.metrics.measurement_method.as_deref(),
            Some(SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK)
        );
    }

    #[test]
    fn older_v0_receipt_without_decode_diagnostics_still_deserializes() {
        let mut receipt = sample_receipt();
        receipt.decode_diagnostics = None;
        let json = ShortAudioReceipt::try_new(receipt)
            .unwrap()
            .to_pretty_json()
            .unwrap();
        assert!(
            !json.contains("decode_diagnostics"),
            "empty diagnostics must stay omitted for older readers"
        );
        let stripped = json
            .lines()
            .filter(|line| !line.contains("decode_diagnostics"))
            .collect::<Vec<_>>()
            .join("\n");
        let loaded = ShortAudioReceipt::from_json_str(&stripped).unwrap();
        assert!(loaded.decode_diagnostics.is_none());
    }
}
