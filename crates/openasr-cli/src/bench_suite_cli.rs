//! `openasr bench-suite` — the committed performance regression gate.
//!
//! Runs each entry in `perf/suite.toml` through the **real** transcription call
//! path (`transcribe_with_backend(Native, ..)`, the same one
//! `transcribe --benchmark` uses), measures RTF + peak RSS + WER, then either writes a new
//! baseline (`--write-baseline`) or compares against a committed baseline and
//! fails closed on regression.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use openasr_core::{
    BackendKind, ExecutionTarget, NATIVE_RUNTIME_MODEL_ID_AUTO, SuiteBaseline, SuiteConfig,
    SuiteEntry, SuiteEntryMetrics, TranscriptionRequest, check_quant_ordering, check_vs_cpp,
    compare_to_baseline, load_config, openasr_home, peak_rss_bytes, prepare_audio_input,
    render_suite_json, render_suite_markdown, validate_local_native_model_pack_path, wer_counts,
};

use crate::cli_args::BenchSuiteCommandOptions;
use crate::native_segment_cli::{
    audio_preparation_options, configure_native_cpu_inference_threads, resolve_explicit_ffmpeg_bin,
    resolve_ffmpeg_bin, transcribe_with_backend,
};

/// Marker prefixing the per-entry metrics JSON a child process emits on stdout,
/// so the parent can extract it regardless of any other stdout from model code.
const ENTRY_IPC_MARKER: &str = "__OASR_BENCH_ENTRY__";
/// Schema version of the parent↔child metrics envelope. Bumped whenever
/// `SuiteEntryMetrics` changes shape; the parent rejects a mismatch so a stale
/// child binary can never silently corrupt gating data with defaulted fields.
const BENCH_ENTRY_IPC_SCHEMA_VERSION: u32 = 1;

pub(crate) fn bench_suite(options: BenchSuiteCommandOptions<'_>) -> Result<()> {
    let config = load_suite_config(options.config)?;
    if config.schema_version != 1 {
        bail!(
            "Unsupported suite schema_version {} (expected 1) in {}",
            config.schema_version,
            options.config.display()
        );
    }

    let home = openasr_home()?;
    let app_config = load_config(&home)?;
    let ffmpeg_bin_explicit =
        resolve_explicit_ffmpeg_bin(options.runtime_paths.ffmpeg_bin.clone(), &app_config)
            .is_some();
    let ffmpeg_bin = resolve_ffmpeg_bin(options.runtime_paths.ffmpeg_bin.clone(), &app_config);

    // Child mode (per-entry subprocess isolation): run exactly one entry in this
    // fresh process and emit its metrics as a JSON envelope. `ru_maxrss` is a
    // process high-water mark, so measuring each entry in its own process is the
    // only way to get an uncontaminated peak RSS (the parent's sequential
    // in-process loop made every entry inherit the largest earlier entry's peak).
    if let Some(entry_id) = options.run_single_entry {
        return run_single_entry_child(
            &config,
            entry_id,
            ffmpeg_bin,
            ffmpeg_bin_explicit,
            options.runs,
        );
    }

    let mut metrics = Vec::new();
    for entry in &config.entries {
        if let Some(family) = options.family
            && entry.family != family
        {
            continue;
        }
        match spawn_entry_subprocess(&options, entry)? {
            Some(measured) => {
                eprintln!(
                    "  ✓ {} ({} {}) — {} ms",
                    entry.id, entry.family, entry.quant, measured.elapsed_ms
                );
                metrics.push(measured);
            }
            None => {
                eprintln!(
                    "  · {} skipped — pack not found: {}",
                    entry.id,
                    entry.pack_path.display()
                );
            }
        }
    }

    if metrics.is_empty() {
        bail!("No suite entries ran (all filtered out or packs missing).");
    }

    if let Some(path) = options.write_baseline {
        let baseline = SuiteBaseline {
            schema_version: 1,
            host_note: host_note(),
            entries: metrics.clone(),
        };
        write_baseline(path, &baseline)?;
        eprintln!("Wrote baseline: {}", path.display());
        print!("{}", render_suite_markdown(&metrics, &[]));
        return Ok(());
    }

    let baseline_path = options
        .baseline
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_baseline_path(options.config));
    let mut baseline = load_baseline(&baseline_path).with_context(|| {
        format!(
            "Could not load baseline {} (run with --write-baseline to create it)",
            baseline_path.display()
        )
    })?;
    // When the run is filtered to one family, only gate against that family's
    // baseline entries — other families were intentionally not run.
    if let Some(family) = options.family {
        baseline.entries.retain(|entry| entry.family == family);
    }

    let mut findings = compare_to_baseline(&metrics, &baseline, &config.default_tolerances);
    findings.extend(check_quant_ordering(&metrics, &config.default_tolerances));
    findings.extend(check_vs_cpp(&metrics, &config.default_tolerances));
    let rendered = render(&metrics, &findings, options.format);
    print!("{rendered}");

    if !findings.is_empty() {
        bail!(
            "Performance regression detected: {} finding(s) exceeded tolerance.",
            findings.len()
        );
    }
    Ok(())
}

/// Run a single entry through the real native backend. Returns `Ok(None)` when
/// the pack is absent and the entry is `optional` (host lacks this pack).
/// The bench passes select the backend via OPENASR_GGML_BACKEND. Map it to an
/// explicit per-request execution target so families whose Auto default
/// differs from the env-global (xasr stays CPU on Auto) actually run what the
/// pass label claims — a metal pass must never silently record CPU numbers.
fn bench_execution_target() -> Option<ExecutionTarget> {
    let raw = std::env::var("OPENASR_GGML_BACKEND").ok()?;
    let value = raw.trim();
    if value.eq_ignore_ascii_case("cpu") {
        Some(ExecutionTarget::Cpu)
    } else if value.is_empty() {
        None
    } else {
        // metal / gpu / vendor aliases: the pass explicitly asked for an
        // accelerated backend.
        Some(ExecutionTarget::Accelerated)
    }
}

/// Builds one suite entry's [`TranscriptionRequest`]. Split out from
/// [`run_entry`] so the `RequestSource` wiring is unit-testable without a
/// real model pack.
fn suite_entry_transcription_request(
    entry: &SuiteEntry,
    validated_pack: &Path,
    prepared: &openasr_core::PreparedAudioInput,
) -> TranscriptionRequest {
    TranscriptionRequest::new(prepared.path(), NATIVE_RUNTIME_MODEL_ID_AUTO)
        .with_source(openasr_core::RequestSource::CliBenchSuite)
        .with_model_pack_path(Some(validated_pack.to_path_buf()))
        .with_language(entry.language.clone())
        .with_task(entry.task)
        .with_execution_target(bench_execution_target())
        // The perf suite measures ASR decode RTF/RSS; an optional
        // post-process stage running when a FireRedPunc pack happens to be
        // installed would skew both, so it stays off for every entry.
        .with_punctuation(false)
        .with_prepared_samples(prepared.shared_samples())
}

fn run_entry(
    entry: &SuiteEntry,
    ffmpeg_bin: Option<PathBuf>,
    ffmpeg_bin_explicit: bool,
    runs: usize,
) -> Result<Option<SuiteEntryMetrics>> {
    if !entry.pack_path.exists() {
        if entry.optional {
            return Ok(None);
        }
        bail!(
            "Required pack not found: {} (mark the entry `optional = true` to skip on this host)",
            entry.pack_path.display()
        );
    }
    let validated_pack = validate_local_native_model_pack_path(&entry.pack_path)
        .with_context(|| format!("Invalid model pack: {}", entry.pack_path.display()))?;

    let prepared = prepare_audio_input(
        &entry.audio_path,
        &audio_preparation_options(BackendKind::Native, ffmpeg_bin, ffmpeg_bin_explicit),
    )
    .with_context(|| format!("Could not prepare audio: {}", entry.audio_path.display()))?;

    let audio_seconds = prepared.duration_seconds();
    let reference = resolve_reference(entry)?;

    // Best-of-N: keep the fastest wall-clock sample so a single noisy run
    // (background load, scheduling) doesn't gate. WER/text are deterministic
    // for a fixed model+audio, so they come from the first run.
    let mut best_elapsed: Option<std::time::Duration> = None;
    let mut min_peak_rss: Option<u64> = None;
    // Cold-vs-warm split (goals 7+8 Step 0): the first run pays the one-time
    // weight-bind/load (prepared-runtime cache is cold); runs >= 1 reuse the hot
    // cache and are compute-only. `load_ms = cold - warm` surfaces the weight-bind
    // cost the best-of-N `elapsed_ms` otherwise hides — the metric the zero-copy
    // binding lever (Step 1+) must move. Per-request-loading families (no cache)
    // show ~0 here, itself a signal. Requires runs >= 2 to split.
    let mut first_run: Option<std::time::Duration> = None;
    let mut warm_best: Option<std::time::Duration> = None;
    let mut transcription_text = String::new();
    let mut segment_count = 0;
    for run_index in 0..runs.max(1) {
        let request = suite_entry_transcription_request(entry, &validated_pack, &prepared);
        configure_native_cpu_inference_threads();
        let started = Instant::now();
        let transcription = transcribe_with_backend(BackendKind::Native, request)?;
        let elapsed = started.elapsed();
        if best_elapsed.is_none_or(|best| elapsed < best) {
            best_elapsed = Some(elapsed);
        }
        if let Some(peak) = peak_rss_bytes() {
            min_peak_rss = Some(min_peak_rss.map_or(peak, |current| current.min(peak)));
        }
        if run_index == 0 {
            first_run = Some(elapsed);
            transcription_text = transcription.text;
            segment_count = transcription.segments.len();
        } else {
            warm_best = Some(warm_best.map_or(elapsed, |best| best.min(elapsed)));
        }
    }
    let elapsed = best_elapsed.unwrap_or_default();
    let (load_ms, compute_ms) = match (first_run, warm_best) {
        (Some(cold), Some(warm)) => (
            Some(cold.saturating_sub(warm).as_millis()),
            Some(warm.as_millis()),
        ),
        _ => (None, None),
    };
    let rtf = audio_seconds
        .filter(|seconds| *seconds > 0.0)
        .map(|seconds| elapsed.as_secs_f64() / seconds);

    // "Beat comparable open source": time the same-model whisper.cpp baseline.
    // whisper.cpp is an external process that needs a real WAV file path --
    // it cannot take the in-memory samples the symphonia decode path now
    // hands back for non-WAV/non-conformant-WAV entries (`prepared.path()`
    // there is only the *original* source file, not a WAV; see
    // `PreparedAudioInput::path`'s doc comment). Every entry in
    // `perf/suite.toml` today is already a conformant WAV (the passthrough
    // path, always a real file), so this only ever skips the comparison for a
    // hypothetical future non-WAV entry rather than handing whisper.cpp a
    // path it cannot read.
    let cpp_best_ms = match (&entry.cpp_binary, &entry.cpp_model) {
        (Some(binary), Some(model))
            if binary.exists() && model.exists() && prepared.samples().is_none() =>
        {
            time_whisper_cpp(binary, model, prepared.path(), runs.max(1))
        }
        _ => None,
    };

    let (wer_value, wer_errors, wer_ref_words) = match reference {
        Some(reference) => {
            let counts = wer_counts(&transcription_text, &reference);
            let value = if counts.ref_units == 0 {
                None
            } else {
                Some(counts.errors as f64 / counts.ref_units as f64)
            };
            (value, Some(counts.errors), Some(counts.ref_units))
        }
        None => (None, None, None),
    };

    let capture_transcription_text =
        std::env::var("OPENASR_BENCH_SUITE_CAPTURE_TRANSCRIPT").is_ok_and(|value| value == "1");

    Ok(Some(SuiteEntryMetrics {
        id: entry.id.clone(),
        family: entry.family.clone(),
        quant: entry.quant.clone(),
        elapsed_ms: elapsed.as_millis(),
        audio_seconds,
        rtf,
        peak_rss_bytes: min_peak_rss,
        wer: wer_value,
        wer_errors,
        wer_ref_words,
        transcription_text: if capture_transcription_text {
            transcription_text.clone()
        } else {
            String::new()
        },
        text_chars: transcription_text.chars().count(),
        segment_count,
        gating: entry.is_gating(),
        ordering_group: entry.ordering_group.clone(),
        cpp_best_ms,
        load_ms,
        compute_ms,
    }))
}

/// Child entrypoint: run exactly one entry (by id) in this fresh process and
/// print its metrics as a marked JSON envelope on stdout. A missing optional
/// pack emits `metrics: null` (the parent treats it as skipped); a hard failure
/// returns an error so the child exits non-zero and the parent surfaces it.
fn run_single_entry_child(
    config: &SuiteConfig,
    entry_id: &str,
    ffmpeg_bin: Option<PathBuf>,
    ffmpeg_bin_explicit: bool,
    runs: usize,
) -> Result<()> {
    let entry = config
        .entries
        .iter()
        .find(|entry| entry.id == entry_id)
        .ok_or_else(|| anyhow::anyhow!("unknown suite entry id '{entry_id}'"))?;
    let measured = run_entry(entry, ffmpeg_bin, ffmpeg_bin_explicit, runs)
        .with_context(|| format!("Suite entry '{}' failed", entry.id))?;
    let envelope = serde_json::json!({
        "schema_version": BENCH_ENTRY_IPC_SCHEMA_VERSION,
        "metrics": measured,
    });
    let line = serde_json::to_string(&envelope).context("Could not serialize entry metrics")?;
    println!("{ENTRY_IPC_MARKER} {line}");
    Ok(())
}

/// Parent side: run one entry in its own subprocess (`openasr bench-suite
/// --run-single-entry <id>`) and parse back its metrics envelope. Isolation is
/// what makes the entry's peak RSS clean; RTF/WER are deterministic and
/// unchanged vs in-process.
fn spawn_entry_subprocess(
    options: &BenchSuiteCommandOptions<'_>,
    entry: &SuiteEntry,
) -> Result<Option<SuiteEntryMetrics>> {
    use std::process::Command;

    let exe = std::env::current_exe()
        .context("Could not resolve current executable for per-entry subprocess")?;
    let mut command = Command::new(&exe);
    command
        .arg("bench-suite")
        .arg("--config")
        .arg(options.config)
        .arg("--runs")
        .arg(options.runs.to_string())
        .arg("--run-single-entry")
        .arg(&entry.id);
    if let Some(ffmpeg_bin) = &options.runtime_paths.ffmpeg_bin {
        command.arg("--ffmpeg-bin").arg(ffmpeg_bin);
    }

    let output = command
        .output()
        .with_context(|| format!("Could not spawn per-entry subprocess for '{}'", entry.id))?;
    if !output.status.success() {
        bail!(
            "Suite entry '{}' subprocess failed (exit {:?}):\n{}",
            entry.id,
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let payload = stdout
        .lines()
        .find_map(|line| line.strip_prefix(ENTRY_IPC_MARKER))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Entry '{}' subprocess produced no metrics envelope",
                entry.id
            )
        })?
        .trim();
    let envelope: serde_json::Value = serde_json::from_str(payload)
        .with_context(|| format!("Could not parse metrics envelope for '{}'", entry.id))?;
    let schema = envelope
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    if schema != Some(u64::from(BENCH_ENTRY_IPC_SCHEMA_VERSION)) {
        bail!(
            "Entry '{}' subprocess metrics schema {:?} != expected {} (stale binary? rebuild)",
            entry.id,
            schema,
            BENCH_ENTRY_IPC_SCHEMA_VERSION
        );
    }
    let metrics: Option<SuiteEntryMetrics> = serde_json::from_value(
        envelope
            .get("metrics")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    )
    .with_context(|| format!("Could not deserialize metrics for '{}'", entry.id))?;
    Ok(metrics)
}

/// Time a same-model `whisper.cpp` (or compatible) CLI on the clip, best-of-N
/// wall, in ms. Returns `None` if it fails to run. Output is suppressed; we time
/// the whole process (model load + compute), matching how the openasr side is
/// measured around `transcribe_with_backend`.
fn time_whisper_cpp(binary: &Path, model: &Path, audio: &Path, runs: usize) -> Option<u128> {
    use std::process::{Command, Stdio};
    let mut best: Option<u128> = None;
    for _ in 0..runs.max(1) {
        let started = Instant::now();
        let status = Command::new(binary)
            .args(["-m"])
            .arg(model)
            .args(["-f"])
            .arg(audio)
            .arg("-nt")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let elapsed = started.elapsed().as_millis();
        match status {
            Ok(s) if s.success() => {
                best = Some(best.map_or(elapsed, |b| b.min(elapsed)));
            }
            _ => return None,
        }
    }
    best
}

fn resolve_reference(entry: &SuiteEntry) -> Result<Option<String>> {
    if let Some(inline) = &entry.reference {
        return Ok(Some(inline.clone()));
    }
    if let Some(path) = &entry.reference_text_path {
        let text = fs::read_to_string(path)
            .with_context(|| format!("Could not read reference: {}", path.display()))?;
        return Ok(Some(text));
    }
    Ok(None)
}

fn render(
    metrics: &[SuiteEntryMetrics],
    findings: &[openasr_core::RegressionFinding],
    format: openasr_core::BenchmarkFormat,
) -> String {
    use openasr_core::BenchmarkFormat;
    match format {
        BenchmarkFormat::Json => {
            serde_json::to_string_pretty(metrics).unwrap_or_else(|error| format!("{error}\n"))
        }
        BenchmarkFormat::Text | BenchmarkFormat::Markdown => {
            render_suite_markdown(metrics, findings)
        }
    }
}

fn load_suite_config(path: &Path) -> Result<SuiteConfig> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("Could not read suite config: {}", path.display()))?;
    let config: SuiteConfig = toml::from_str(&text)
        .with_context(|| format!("Invalid suite config TOML: {}", path.display()))?;
    Ok(config)
}

fn load_baseline(path: &Path) -> Result<SuiteBaseline> {
    let text = fs::read_to_string(path)?;
    let baseline: SuiteBaseline = serde_json::from_str(&text)
        .with_context(|| format!("Invalid baseline JSON: {}", path.display()))?;
    Ok(baseline)
}

fn write_baseline(path: &Path, baseline: &SuiteBaseline) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let json = render_suite_json(baseline).context("Could not serialize baseline")?;
    fs::write(path, json)
        .with_context(|| format!("Could not write baseline: {}", path.display()))?;
    Ok(())
}

fn default_baseline_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("perf"))
        .join("baselines")
        .join(format!("{}.json", host_slug()))
}

fn host_note() -> String {
    format!("{} / {}", std::env::consts::OS, std::env::consts::ARCH)
}

fn host_slug() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_wav_fixture_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist")
    }

    fn sample_suite_entry() -> SuiteEntry {
        SuiteEntry {
            id: "test-entry".to_string(),
            family: "whisper".to_string(),
            quant: "q4_k".to_string(),
            pack_path: PathBuf::from("/nonexistent/pack.oasr"),
            audio_path: sample_wav_fixture_path(),
            language: None,
            task: None,
            reference: None,
            reference_text_path: None,
            optional: true,
            gating: None,
            ordering_group: None,
            cpp_binary: None,
            cpp_model: None,
        }
    }

    // Regression guard for the `openasr bench-suite` entry point: its
    // `TranscriptionRequest` must log `RequestSource::CliBenchSuite`, distinct
    // from `transcribe --benchmark`'s `CliTranscribe` -- see
    // `RequestSource::CliBenchSuite`'s doc comment for why the two CLI timing
    // paths must not collapse to the same `daemon.log` label.
    #[test]
    fn suite_entry_transcription_request_labels_source_as_cli_bench_suite() {
        let entry = sample_suite_entry();
        let prepared = prepare_audio_input(
            &entry.audio_path,
            &audio_preparation_options(BackendKind::Native, None, false),
        )
        .expect("fixture wav must prepare");
        let validated_pack = PathBuf::from("/nonexistent/pack.oasr");
        let request = suite_entry_transcription_request(&entry, &validated_pack, &prepared);
        assert_eq!(request.source, openasr_core::RequestSource::CliBenchSuite);
    }
}
