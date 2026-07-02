//! Committed performance-suite config + baseline schema and the regression
//! comparison logic. Pure data and comparison — audio I/O and the actual
//! transcription run live in the CLI, which reuses the real backend call path.
//!
//! A fixed audio set at a fixed quantization, one machine-readable baseline
//! per host profile, and a gate that fails closed on regression (RTF/peak-RSS
//! relative, WER absolute).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Regression thresholds. RTF and peak-RSS are compared relative (a fraction of
/// the baseline); WER is compared as an absolute delta because it is already a
/// ratio.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct Tolerances {
    /// Max allowed relative RTF increase (e.g. `0.25` = +25%). RTF is a
    /// single-run wall-clock measurement, so the default is generous to avoid
    /// false positives from warmup/scheduling jitter on short clips.
    pub rtf_rel: f64,
    /// Max allowed relative peak-RSS increase, applied only when
    /// `gate_peak_rss` is set.
    pub peak_rss_rel: f64,
    /// Max allowed absolute WER increase (e.g. `0.02`). WER is deterministic
    /// for a fixed model+audio, so this is the primary, reliable gate.
    pub wer_abs: f64,
    /// Whether peak-RSS regressions fail the gate. Off by default for ad-hoc
    /// suites; committed perf runs execute each entry in a fresh CLI subprocess,
    /// so `perf/suite.toml` enables it for the stable gating entries.
    #[serde(default)]
    pub gate_peak_rss: bool,
    /// Max fraction by which openasr may be slower than a same-model
    /// `whisper.cpp` baseline before the "beat comparable open source" gate
    /// fails (e.g. `0.05` = openasr within +5% of whisper.cpp). Only applies to
    /// entries with a `cpp_*` baseline, and only when `gate_vs_cpp` is set.
    #[serde(default = "default_cpp_slack")]
    pub cpp_slack: f64,
    /// Whether the vs-whisper.cpp comparison fails the gate. Off by default for
    /// ad-hoc suites; committed perf runs use per-entry subprocess isolation and
    /// opt in once a matched-thread same-model baseline is recorded.
    #[serde(default)]
    pub gate_vs_cpp: bool,
}

fn default_cpp_slack() -> f64 {
    0.05
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            rtf_rel: 0.25,
            peak_rss_rel: 0.20,
            wer_abs: 0.02,
            gate_peak_rss: false,
            cpp_slack: default_cpp_slack(),
            gate_vs_cpp: false,
        }
    }
}

/// One suite entry: a (family, quant) measured by running `pack_path` on
/// `audio_path` and scoring against `reference`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SuiteEntry {
    pub id: String,
    pub family: String,
    pub quant: String,
    pub pack_path: PathBuf,
    pub audio_path: PathBuf,
    /// Source language hint (e.g. `fr`); omit for the model default. Lets a
    /// non-English transcribe / translate entry exercise the multilingual path.
    #[serde(default)]
    pub language: Option<String>,
    /// Speech task (`transcribe` | `translate`); omit for the default transcribe.
    #[serde(default)]
    pub task: Option<crate::TranscriptionTask>,
    /// Inline reference transcript, or a path to one (`reference_text_path`).
    #[serde(default)]
    pub reference: Option<String>,
    #[serde(default)]
    pub reference_text_path: Option<PathBuf>,
    /// `true` keeps a missing pack non-fatal (entry skipped) so the suite still
    /// runs on hosts that lack this particular pack.
    #[serde(default)]
    pub optional: bool,
    /// Non-gating entries are measured and reported but never fail the gate
    /// (e.g. families with an unverified pack at this quant).
    #[serde(default)]
    pub gating: Option<bool>,
    /// Entries sharing an `ordering_group` are the same model at different
    /// quantizations; the suite asserts their RTF is ordered by quant rank
    /// (q4_k ≤ q8_0 ≤ fp16 — more compression should not be slower).
    #[serde(default)]
    pub ordering_group: Option<String>,
    /// `whisper.cpp` (or compatible) CLI binary to time on the same clip as a
    /// "beat comparable open source" baseline. When set with `cpp_model`, the
    /// suite runs it best-of-N and gates that openasr is not slower beyond
    /// `cpp_slack`.
    #[serde(default)]
    pub cpp_binary: Option<PathBuf>,
    /// The ggml model the `cpp_binary` should load (same model/quant as the
    /// openasr `pack_path`, for a fair comparison).
    #[serde(default)]
    pub cpp_model: Option<PathBuf>,
}

impl SuiteEntry {
    pub fn is_gating(&self) -> bool {
        self.gating.unwrap_or(true)
    }
}

/// Top-level committed suite configuration (`perf/suite.toml`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SuiteConfig {
    pub schema_version: u32,
    #[serde(default)]
    pub default_tolerances: Tolerances,
    pub entries: Vec<SuiteEntry>,
}

/// Measured metrics for one entry. Optional fields are `None` when the
/// measurement is unavailable on the host (e.g. peak-RSS on Windows, or RTF
/// when the audio duration could not be probed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuiteEntryMetrics {
    pub id: String,
    pub family: String,
    pub quant: String,
    pub elapsed_ms: u128,
    pub audio_seconds: Option<f64>,
    pub rtf: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub wer: Option<f64>,
    pub wer_errors: Option<usize>,
    pub wer_ref_words: Option<usize>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub transcription_text: String,
    pub text_chars: usize,
    pub segment_count: usize,
    #[serde(default)]
    pub gating: bool,
    #[serde(default)]
    pub ordering_group: Option<String>,
    /// Best-of-N wall time of the same-model `whisper.cpp` baseline, in ms, when
    /// a `cpp_*` baseline was configured. `None` if no comparison was run.
    #[serde(default)]
    pub cpp_best_ms: Option<u128>,
    /// Cold-vs-warm split from best-of-N (goals 7+8 Step 0). `compute_ms` is the
    /// warm steady-state wall (min over runs >= 1, prepared-runtime cache hot);
    /// `load_ms` is the first-run weight-bind/load cost (cold first run minus
    /// warm). Both `None` when fewer than 2 runs were taken (cannot split).
    /// Families that reload per request (no prepared-runtime cache) show
    /// `load_ms` ~ 0 — load is amortized into every run, itself a signal.
    #[serde(default)]
    pub load_ms: Option<u128>,
    #[serde(default)]
    pub compute_ms: Option<u128>,
}

/// A committed baseline: one snapshot of metrics per host profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SuiteBaseline {
    pub schema_version: u32,
    pub host_note: String,
    pub entries: Vec<SuiteEntryMetrics>,
}

impl SuiteBaseline {
    pub fn find(&self, id: &str) -> Option<&SuiteEntryMetrics> {
        self.entries.iter().find(|entry| entry.id == id)
    }
}

/// What kind of regression a finding represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RegressionKind {
    RtfSlower,
    PeakRssHigher,
    WerWorse,
    MissingBaseline,
    MissingCandidate,
    QuantOrderViolation,
    SlowerThanCpp,
}

impl RegressionKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::RtfSlower => "RTF slower",
            Self::PeakRssHigher => "peak RSS higher",
            Self::WerWorse => "WER worse",
            Self::MissingBaseline => "no baseline entry",
            Self::MissingCandidate => "no candidate entry",
            Self::QuantOrderViolation => "quant order violated (more compression slower)",
            Self::SlowerThanCpp => "slower than whisper.cpp (lost 'beat comparable OSS')",
        }
    }
}

/// Rank a quantization by expected speed/size: lower rank = more compressed =
/// expected faster + smaller. Unknown quants return `None` (excluded from the
/// ordering check).
pub fn quant_rank(quant: &str) -> Option<u32> {
    match quant.to_ascii_lowercase().as_str() {
        "q3" | "q3_k" => Some(0),
        "q4_0" | "q4_k" | "q4_k_m" | "q4_k_s" => Some(0),
        "q5_0" | "q5_k" | "q5_k_m" => Some(1),
        "q8_0" => Some(2),
        "f16" | "fp16" => Some(3),
        "f32" | "fp32" => Some(4),
        _ => None,
    }
}

/// Assert, per `ordering_group`, that RTF is non-decreasing in quant rank —
/// i.e. a more-compressed quant is not slower than a less-compressed one beyond
/// a slack tolerance (`rtf_rel`). A violation means the quant pipeline produced
/// a pack that breaks the "smaller quant ⇒ faster" expectation (ROADMAP P2).
pub fn check_quant_ordering(
    metrics: &[SuiteEntryMetrics],
    tolerances: &Tolerances,
) -> Vec<RegressionFinding> {
    use std::collections::BTreeMap;

    let mut groups: BTreeMap<&str, Vec<&SuiteEntryMetrics>> = BTreeMap::new();
    for entry in metrics {
        if let Some(group) = entry.ordering_group.as_deref() {
            groups.entry(group).or_default().push(entry);
        }
    }

    let mut findings = Vec::new();
    for (_group, mut members) in groups {
        members.sort_by_key(|entry| quant_rank(&entry.quant).unwrap_or(u32::MAX));
        // Walk adjacent (more-compressed, less-compressed) pairs by rank and
        // assert RTF does not increase as compression increases.
        for window in members.windows(2) {
            let (faster, slower) = (window[0], window[1]);
            let (Some(rank_a), Some(rank_b)) =
                (quant_rank(&faster.quant), quant_rank(&slower.quant))
            else {
                continue;
            };
            if rank_a >= rank_b {
                continue; // same rank or unordered; skip
            }
            if let (Some(rtf_more_compressed), Some(rtf_less_compressed)) = (faster.rtf, slower.rtf)
            {
                // More-compressed (faster) RTF should be <= less-compressed RTF.
                // Allow rtf_rel slack against the less-compressed reference.
                if rtf_less_compressed > 0.0
                    && (rtf_more_compressed - rtf_less_compressed) / rtf_less_compressed
                        > tolerances.rtf_rel
                {
                    findings.push(RegressionFinding {
                        id: format!("{} > {}", faster.id, slower.id),
                        kind: RegressionKind::QuantOrderViolation,
                        baseline: rtf_less_compressed,
                        candidate: rtf_more_compressed,
                        tolerance: tolerances.rtf_rel,
                    });
                }
            }
        }
    }
    findings
}

/// "Beat comparable open source" gate: for entries with a same-model
/// `whisper.cpp` baseline (`cpp_best_ms`), assert openasr's best wall time is
/// not slower than whisper.cpp beyond `cpp_slack`. This is the committed,
/// regression-defended form of goal 3 ("performance beats comparable OSS").
pub fn check_vs_cpp(
    metrics: &[SuiteEntryMetrics],
    tolerances: &Tolerances,
) -> Vec<RegressionFinding> {
    let mut findings = Vec::new();
    if !tolerances.gate_vs_cpp {
        return findings;
    }
    for entry in metrics {
        let Some(cpp_ms) = entry.cpp_best_ms else {
            continue;
        };
        if cpp_ms == 0 {
            continue;
        }
        let openasr_ms = entry.elapsed_ms;
        let cpp = cpp_ms as f64;
        if (openasr_ms as f64 - cpp) / cpp > tolerances.cpp_slack {
            findings.push(RegressionFinding {
                id: entry.id.clone(),
                kind: RegressionKind::SlowerThanCpp,
                baseline: cpp,
                candidate: openasr_ms as f64,
                tolerance: tolerances.cpp_slack,
            });
        }
    }
    findings
}

/// One regression against the committed baseline.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RegressionFinding {
    pub id: String,
    pub kind: RegressionKind,
    pub baseline: f64,
    pub candidate: f64,
    pub tolerance: f64,
}

/// Compare candidate metrics against a baseline. Gating entries that regress
/// beyond tolerance (or are missing on either side) yield findings; non-gating
/// entries are never reported. Fail-closed: a gating candidate with no baseline
/// counterpart, or a baseline entry with no candidate, is itself a finding.
pub fn compare_to_baseline(
    candidate: &[SuiteEntryMetrics],
    baseline: &SuiteBaseline,
    tolerances: &Tolerances,
) -> Vec<RegressionFinding> {
    let mut findings = Vec::new();

    for metrics in candidate {
        if !metrics.gating {
            continue;
        }
        let Some(base) = baseline.find(&metrics.id) else {
            findings.push(RegressionFinding {
                id: metrics.id.clone(),
                kind: RegressionKind::MissingBaseline,
                baseline: 0.0,
                candidate: 0.0,
                tolerance: 0.0,
            });
            continue;
        };

        if let (Some(base_rtf), Some(cand_rtf)) = (base.rtf, metrics.rtf)
            && exceeds_relative(base_rtf, cand_rtf, tolerances.rtf_rel)
        {
            findings.push(RegressionFinding {
                id: metrics.id.clone(),
                kind: RegressionKind::RtfSlower,
                baseline: base_rtf,
                candidate: cand_rtf,
                tolerance: tolerances.rtf_rel,
            });
        }

        if tolerances.gate_peak_rss
            && let (Some(base_rss), Some(cand_rss)) = (base.peak_rss_bytes, metrics.peak_rss_bytes)
            && exceeds_relative(base_rss as f64, cand_rss as f64, tolerances.peak_rss_rel)
        {
            findings.push(RegressionFinding {
                id: metrics.id.clone(),
                kind: RegressionKind::PeakRssHigher,
                baseline: base_rss as f64,
                candidate: cand_rss as f64,
                tolerance: tolerances.peak_rss_rel,
            });
        }

        if let (Some(base_wer), Some(cand_wer)) = (base.wer, metrics.wer)
            && cand_wer - base_wer > tolerances.wer_abs
        {
            findings.push(RegressionFinding {
                id: metrics.id.clone(),
                kind: RegressionKind::WerWorse,
                baseline: base_wer,
                candidate: cand_wer,
                tolerance: tolerances.wer_abs,
            });
        }
    }

    // A *gating* baseline entry with no candidate is a silently-dropped entry
    // (fail-closed). Non-gating entries, and entries the caller filtered out
    // (e.g. via --family), are not required to be present.
    for base in &baseline.entries {
        if base.gating && !candidate.iter().any(|m| m.id == base.id) {
            findings.push(RegressionFinding {
                id: base.id.clone(),
                kind: RegressionKind::MissingCandidate,
                baseline: 0.0,
                candidate: 0.0,
                tolerance: 0.0,
            });
        }
    }

    findings
}

fn exceeds_relative(baseline: f64, candidate: f64, tol_rel: f64) -> bool {
    if baseline <= 0.0 {
        return false;
    }
    (candidate - baseline) / baseline > tol_rel
}

fn format_mb(bytes: Option<u64>) -> String {
    bytes.map_or_else(
        || "—".to_string(),
        |value| format!("{:.0}", value as f64 / (1024.0 * 1024.0)),
    )
}

fn format_rtf(value: Option<f64>) -> String {
    value.map_or_else(|| "—".to_string(), |value| format!("{value:.4}"))
}

fn format_wer(value: Option<f64>) -> String {
    value.map_or_else(|| "—".to_string(), |value| format!("{:.2}%", value * 100.0))
}

fn format_ms(value: Option<u128>) -> String {
    value.map_or_else(|| "—".to_string(), |value| format!("{value}"))
}

/// Render a Markdown metrics table plus any regression findings.
pub fn render_suite_markdown(
    metrics: &[SuiteEntryMetrics],
    findings: &[RegressionFinding],
) -> String {
    let mut out = String::from(
        "| Entry | Family | Quant | RTF | Peak MB | Load ms | Compute ms | WER | Segments | Gate |\n| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n",
    );
    for entry in metrics {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            entry.id,
            entry.family,
            entry.quant,
            format_rtf(entry.rtf),
            format_mb(entry.peak_rss_bytes),
            format_ms(entry.load_ms),
            format_ms(entry.compute_ms),
            format_wer(entry.wer),
            entry.segment_count,
            if entry.gating { "gate" } else { "info" },
        ));
    }

    // "Beat comparable open source" comparisons, shown on pass too.
    let cpp_lines: Vec<String> = metrics
        .iter()
        .filter_map(|entry| {
            let cpp = entry.cpp_best_ms?;
            if cpp == 0 {
                return None;
            }
            let ratio = entry.elapsed_ms as f64 / cpp as f64;
            let verdict = if ratio < 1.0 {
                format!("openasr FASTER by {:.1}%", (1.0 - ratio) * 100.0)
            } else {
                format!("openasr slower by {:.1}%", (ratio - 1.0) * 100.0)
            };
            Some(format!(
                "- `{}`: openasr {} ms vs whisper.cpp {} ms — {}",
                entry.id, entry.elapsed_ms, cpp, verdict
            ))
        })
        .collect();
    if !cpp_lines.is_empty() {
        out.push_str("\n**vs whisper.cpp (same model):**\n");
        out.push_str(&cpp_lines.join("\n"));
        out.push('\n');
    }

    if findings.is_empty() {
        out.push_str("\n**Regression gate: PASS** (no gating entry exceeded tolerance).\n");
    } else {
        out.push_str(&format!(
            "\n**Regression gate: FAIL** ({} finding(s)):\n",
            findings.len()
        ));
        for finding in findings {
            out.push_str(&format!(
                "- `{}` — {}: baseline {:.4} -> candidate {:.4} (tolerance {:.4})\n",
                finding.id,
                finding.kind.label(),
                finding.baseline,
                finding.candidate,
                finding.tolerance,
            ));
        }
    }
    out
}

/// Serialize a baseline to pretty JSON (for `--write-baseline`).
pub fn render_suite_json(baseline: &SuiteBaseline) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(baseline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(id: &str, rtf: f64, rss: u64, wer: f64, gating: bool) -> SuiteEntryMetrics {
        SuiteEntryMetrics {
            id: id.to_string(),
            family: "whisper".to_string(),
            quant: "q8_0".to_string(),
            elapsed_ms: 100,
            audio_seconds: Some(10.0),
            rtf: Some(rtf),
            peak_rss_bytes: Some(rss),
            wer: Some(wer),
            wer_errors: Some(0),
            wer_ref_words: Some(10),
            transcription_text: String::new(),
            text_chars: 42,
            segment_count: 1,
            gating,
            ordering_group: None,
            cpp_best_ms: None,
            load_ms: None,
            compute_ms: None,
        }
    }

    fn ordering_metric(id: &str, quant: &str, rtf: f64) -> SuiteEntryMetrics {
        let mut m = metric(id, rtf, 1_000, 0.0, false);
        m.quant = quant.to_string();
        m.ordering_group = Some("g".to_string());
        m
    }

    fn baseline_with(entry: SuiteEntryMetrics) -> SuiteBaseline {
        SuiteBaseline {
            schema_version: 1,
            host_note: "test".to_string(),
            entries: vec![entry],
        }
    }

    #[test]
    fn suite_json_omits_empty_transcription_text() {
        let json =
            render_suite_json(&baseline_with(metric("entry", 0.1, 1_000, 0.0, false))).unwrap();

        assert!(!json.contains("transcription_text"));
    }

    #[test]
    fn suite_json_includes_captured_transcription_text() {
        let mut entry = metric("entry", 0.1, 1_000, 0.0, false);
        entry.transcription_text = "ask not what your country can do for you".to_string();

        let json = render_suite_json(&baseline_with(entry)).unwrap();

        assert!(json.contains("\"transcription_text\""));
        assert!(json.contains("ask not what your country can do for you"));
    }

    #[test]
    fn quant_ordering_holds_when_more_compression_is_faster() {
        let metrics = vec![
            ordering_metric("fp16", "fp16", 0.30),
            ordering_metric("q4k", "q4_k", 0.10),
            ordering_metric("q8", "q8_0", 0.20),
        ];
        let findings = check_quant_ordering(&metrics, &Tolerances::default());
        assert!(
            findings.is_empty(),
            "ordered RTF must not flag: {findings:?}"
        );
    }

    #[test]
    fn quant_ordering_flags_when_more_compression_is_slower() {
        // q4_k (most compressed) is much SLOWER than fp16 — a violation.
        let metrics = vec![
            ordering_metric("fp16", "fp16", 0.10),
            ordering_metric("q4k", "q4_k", 0.30),
        ];
        let findings = check_quant_ordering(&metrics, &Tolerances::default());
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, RegressionKind::QuantOrderViolation);
    }

    #[test]
    fn quant_ordering_ignores_entries_without_group() {
        let metrics = vec![metric("a", 0.5, 1_000, 0.0, false)];
        assert!(check_quant_ordering(&metrics, &Tolerances::default()).is_empty());
    }

    #[test]
    fn committed_suite_covers_quant_ordering_groups() {
        let suite_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../perf/suite.toml");
        let suite: SuiteConfig =
            toml::from_str(&std::fs::read_to_string(suite_path).expect("read perf suite"))
                .expect("parse perf suite");

        let group = |id: &str| {
            suite
                .entries
                .iter()
                .find(|entry| entry.id == id)
                .unwrap_or_else(|| panic!("missing perf suite entry {id}"))
                .ordering_group
                .as_deref()
        };

        for id in [
            "qwen3-asr-0.6b-q3k",
            "qwen3-asr-0.6b-q4k",
            "qwen3-asr-0.6b-q8",
            "qwen3-asr-0.6b-fp16",
        ] {
            assert_eq!(group(id), Some("qwen-0.6b"));
        }

        for id in ["cohere-transcribe-q8", "cohere-transcribe-q4k"] {
            assert_eq!(group(id), Some("cohere-transcribe"));
        }

        for id in [
            "whisper-small-en-q4k",
            "whisper-small-en-q8",
            "whisper-small-en-fp16",
        ] {
            assert_eq!(group(id), Some("whisper-small-en"));
        }

        assert_eq!(group("parakeet-ctc-0.6b-q8"), Some("parakeet-ctc-0.6b"));
        assert_eq!(group("parakeet-ctc-0.6b-fp16"), Some("parakeet-ctc-0.6b"));
        assert_eq!(
            group("parakeet-ctc-0.6b-q4k"),
            None,
            "parakeet q4_k is intentionally excluded because its size win is not an RTF win"
        );
    }

    #[test]
    fn quant_rank_orders_compression() {
        assert!(quant_rank("q4_k") < quant_rank("q8_0"));
        assert!(quant_rank("q8_0") < quant_rank("fp16"));
        assert_eq!(quant_rank("unknown"), None);
    }

    fn gating_cpp_tolerances() -> Tolerances {
        Tolerances {
            gate_vs_cpp: true,
            ..Tolerances::default()
        }
    }

    #[test]
    fn vs_cpp_passes_when_openasr_faster() {
        let mut m = metric("turbo", 0.4, 1_000, 0.0, false);
        m.elapsed_ms = 2380;
        m.cpp_best_ms = Some(2587); // openasr 8% faster
        assert!(check_vs_cpp(&[m], &gating_cpp_tolerances()).is_empty());
    }

    #[test]
    fn vs_cpp_flags_when_openasr_slower_beyond_slack() {
        let mut m = metric("turbo", 0.4, 1_000, 0.0, false);
        m.elapsed_ms = 3000;
        m.cpp_best_ms = Some(2500); // openasr 20% slower, > 5% slack
        let findings = check_vs_cpp(&[m], &gating_cpp_tolerances());
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, RegressionKind::SlowerThanCpp);
    }

    #[test]
    fn vs_cpp_off_by_default_is_informational() {
        let mut m = metric("turbo", 0.4, 1_000, 0.0, false);
        m.elapsed_ms = 9999;
        m.cpp_best_ms = Some(2500); // way slower, but default gate is off
        assert!(check_vs_cpp(&[m], &Tolerances::default()).is_empty());
    }

    #[test]
    fn vs_cpp_skips_entries_without_baseline() {
        let m = metric("a", 0.4, 1_000, 0.0, false); // cpp_best_ms None
        assert!(check_vs_cpp(&[m], &gating_cpp_tolerances()).is_empty());
    }

    fn baseline(entries: Vec<SuiteEntryMetrics>) -> SuiteBaseline {
        SuiteBaseline {
            schema_version: 1,
            host_note: "test".to_string(),
            entries,
        }
    }

    #[test]
    fn within_tolerance_passes() {
        let base = baseline(vec![metric("a", 0.10, 1_000, 0.05, true)]);
        let cand = vec![metric("a", 0.11, 1_050, 0.05, true)];
        let findings = compare_to_baseline(&cand, &base, &Tolerances::default());
        assert!(findings.is_empty());
    }

    #[test]
    fn rtf_regression_flagged() {
        let base = baseline(vec![metric("a", 0.10, 1_000, 0.05, true)]);
        let cand = vec![metric("a", 0.20, 1_000, 0.05, true)];
        let findings = compare_to_baseline(&cand, &base, &Tolerances::default());
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, RegressionKind::RtfSlower);
    }

    #[test]
    fn wer_regression_uses_absolute_delta() {
        let base = baseline(vec![metric("a", 0.10, 1_000, 0.05, true)]);
        let cand = vec![metric("a", 0.10, 1_000, 0.10, true)]; // +0.05 abs > 0.02
        let findings = compare_to_baseline(&cand, &base, &Tolerances::default());
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, RegressionKind::WerWorse);
    }

    #[test]
    fn non_gating_entry_never_flags() {
        let base = baseline(vec![metric("a", 0.10, 1_000, 0.05, false)]);
        let cand = vec![metric("a", 0.50, 9_999, 0.90, false)];
        let findings = compare_to_baseline(&cand, &base, &Tolerances::default());
        assert!(findings.is_empty());
    }

    #[test]
    fn missing_baseline_for_gating_entry_is_a_finding() {
        let base = baseline(vec![]);
        let cand = vec![metric("a", 0.10, 1_000, 0.05, true)];
        let findings = compare_to_baseline(&cand, &base, &Tolerances::default());
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].kind, RegressionKind::MissingBaseline);
    }
}
