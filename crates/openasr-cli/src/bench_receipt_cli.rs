//! `openasr bench-receipt short-audio`  -  emit a machine-readable short-audio
//! audit receipt (`openasr.short-audio-receipt.v0`).
//!
//! Explicit tooling command: not part of the default `transcribe` user path.
//! Reuses the same transcription ingress as `transcribe --benchmark`.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, Result, bail};
use openasr_core::{
    BackendKind, ExecutionTarget, GgmlExecutionPlacementSummary, GgmlExecutionTelemetryCollector,
    InstalledPack, NativeExecutionServices, SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK,
    SHORT_AUDIO_RECEIPT_SCHEMA, ShortAudioReceipt, ShortAudioReceiptAudio,
    ShortAudioReceiptMetrics, ShortAudioReceiptPack, ShortAudioReceiptRun,
    ShortAudioReceiptTranscript, TranscriptionRequest, atomic_write_text, list_installed_packs,
    load_config, openasr_home, parse_model_ref, prepare_audio_input, process_memory_snapshot,
    receipt_os_id, resolve_core_commit, resolve_installed_pack_reference, sha256_file,
    validate_local_native_model_pack_path,
};

use crate::cli_args::RuntimePathOverrides;
use crate::native_segment_cli::{prepare_backend_run, transcribe_with_backend};

/// CLI inputs for one short-audio receipt run.
#[derive(Debug, Clone)]
pub(crate) struct ShortAudioReceiptOptions<'a> {
    pub(crate) model: Option<&'a str>,
    pub(crate) audio: &'a Path,
    pub(crate) backend_kind: BackendKind,
    pub(crate) device: &'a str,
    pub(crate) model_pack: Option<&'a Path>,
    pub(crate) out: &'a Path,
    pub(crate) runs: usize,
    pub(crate) warmup_runs: usize,
    pub(crate) core_commit: Option<&'a str>,
    pub(crate) scope: &'a str,
    pub(crate) ffmpeg_bin: Option<PathBuf>,
    pub(crate) git_cwd: Option<&'a Path>,
}

pub(crate) fn bench_receipt_short_audio(
    native_execution_services: &Arc<NativeExecutionServices>,
    options: ShortAudioReceiptOptions<'_>,
) -> Result<()> {
    if options.runs == 0 {
        bail!("--runs must be >= 1");
    }
    if !options.audio.is_file() {
        bail!(
            "Audio file not found: {}\nPass an existing WAV/audio path via --audio.",
            options.audio.display()
        );
    }

    let home = openasr_home()?;
    let config = load_config(&home)?;
    let execution_target = parse_receipt_device(options.device)?;
    let device_label = normalize_device_label(options.device);

    let prepared_run = prepare_backend_run(
        "bench-receipt short-audio",
        options.model,
        Some(options.backend_kind),
        &RuntimePathOverrides {
            ffmpeg_bin: options.ffmpeg_bin.clone(),
        },
        options.model_pack,
        &config,
    )?;

    // Native without a resolved pack cannot bind content bytes; fail closed.
    if prepared_run.backend_kind == BackendKind::Native
        && prepared_run.model_source.model_pack_path.is_none()
    {
        bail!(
            "Native short-audio receipt requires an installed model pack or --model-pack.\nRun: openasr pull <model>   or pass --model-pack <path.oasr>"
        );
    }

    let pack_binding = resolve_pack_binding(
        options.model,
        prepared_run.model_source.model_id.as_str(),
        prepared_run.model_source.model_pack_path.as_deref(),
        prepared_run.backend_kind,
        &home,
    )?;

    let (_audio_size, audio_sha256) = sha256_file(options.audio)
        .with_context(|| format!("Could not hash audio file {}", options.audio.display()))?;

    let prepared_audio = prepare_audio_input(
        options.audio,
        &crate::native_segment_cli::audio_preparation_options(
            prepared_run.backend_kind,
            prepared_run.ffmpeg_bin.clone(),
            prepared_run.ffmpeg_bin_explicit,
        ),
    )?;
    let audio_duration_s = prepared_audio.duration_seconds();
    let memory_before_model = process_memory_snapshot();

    let core_commit = resolve_core_commit(options.core_commit, options.git_cwd).context(
        "Could not resolve core_commit. Pass --core-commit <40-hex>, set OPENASR_BUILD_COMMIT, or run inside a git checkout.",
    )?;

    let total_passes = options.warmup_runs.saturating_add(options.runs);
    let execution_telemetry = GgmlExecutionTelemetryCollector::new();
    let _execution_telemetry_guard = execution_telemetry.install();
    let mut rtf_samples = Vec::with_capacity(options.runs);
    let mut last_text = String::new();
    let mut notes = Vec::new();

    if options.backend_kind == BackendKind::Mock {
        notes.push(
            "backend=mock: transcript and RTF are plumbing-only, not a quality/perf claim"
                .to_string(),
        );
    }
    notes.push(
        "placement is the requested device; observed_placement reports actual ggml graph-node backends"
            .to_string(),
    );
    notes.push(format!(
        "measurement_method={SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK}"
    ));

    for pass in 0..total_passes {
        let is_warmup = pass < options.warmup_runs;
        let request = TranscriptionRequest::new(
            prepared_audio.path(),
            prepared_run.model_source.model_id.clone(),
        )
        .with_source(openasr_core::RequestSource::CliTranscribe)
        .with_model_pack_path(prepared_run.model_source.model_pack_path.clone())
        .with_execution_target(Some(execution_target))
        .with_punctuation(false)
        .with_prepared_samples(prepared_audio.shared_samples())
        .with_display_file_name(
            options
                .audio
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string),
        );

        let started = Instant::now();
        let transcription = transcribe_with_backend(
            native_execution_services,
            prepared_run.backend_kind,
            request,
        )
        .with_context(|| {
            format!(
                "short-audio receipt transcription failed on pass {}/{}",
                pass + 1,
                total_passes
            )
        })?;
        let elapsed = started.elapsed();
        last_text = transcription.text;

        if is_warmup {
            continue;
        }

        let rtf = audio_duration_s
            .filter(|duration| *duration > 0.0)
            .map(|duration| elapsed.as_secs_f64() / duration);
        match rtf {
            Some(value) => rtf_samples.push(value),
            None => notes.push(format!(
                "pass {}: audio duration unavailable; RTF sample omitted",
                pass + 1
            )),
        }
    }

    let warmup = if options.warmup_runs > 0 {
        "warm"
    } else {
        "cold"
    };
    let cache_state = if options.warmup_runs > 0 {
        "populated"
    } else {
        "empty"
    };

    let command = build_command_argv(&options, &pack_binding, &device_label);
    let env_allowlist = capture_env_allowlist();

    let memory_after_model = process_memory_snapshot();
    let observed_placement = execution_telemetry.snapshot();
    if options.backend_kind == BackendKind::Native {
        validate_observed_accelerator_placement(&device_label, &observed_placement)?;
    }
    let receipt = ShortAudioReceipt::try_new(ShortAudioReceipt {
        schema: SHORT_AUDIO_RECEIPT_SCHEMA.to_string(),
        core_commit,
        pack: ShortAudioReceiptPack {
            model_id: pack_binding.model_id,
            content_sha256: pack_binding.content_sha256,
            size_bytes: pack_binding.size_bytes,
            quant: pack_binding.quant,
        },
        audio: ShortAudioReceiptAudio {
            path_or_label: options.audio.display().to_string(),
            sha256: audio_sha256,
            duration_s: audio_duration_s,
        },
        run: ShortAudioReceiptRun {
            backend: prepared_run.backend_kind.to_string(),
            device: device_label.clone(),
            os: receipt_os_id().to_string(),
            command,
            env_allowlist,
            warmup: warmup.to_string(),
            cache_state: cache_state.to_string(),
        },
        metrics: ShortAudioReceiptMetrics {
            wer_or_cer: None,
            rtf_samples,
            rtf_median: None,
            ttft_s: None,
            peak_rss_bytes: memory_after_model.peak_rss_bytes,
            peak_rss_before_model_bytes: memory_before_model.peak_rss_bytes,
            rss_before_model_bytes: memory_before_model.current_rss_bytes,
            rss_after_model_bytes: memory_after_model.current_rss_bytes,
            phys_footprint_before_model_bytes: memory_before_model.current_phys_footprint_bytes,
            phys_footprint_after_model_bytes: memory_after_model.current_phys_footprint_bytes,
            peak_phys_footprint_before_model_bytes: memory_before_model.peak_phys_footprint_bytes,
            peak_phys_footprint_bytes: memory_after_model.peak_phys_footprint_bytes,
            peak_vram_bytes: None,
            measurement_method: Some(SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK.to_string()),
        },
        transcript: ShortAudioReceiptTranscript::from_text(last_text),
        placement: device_label,
        observed_placement: (!observed_placement.is_empty()).then_some(observed_placement),
        scope: options.scope.to_string(),
        notes,
    })
    .context("Constructed short-audio receipt failed validation")?;

    let json = receipt
        .to_pretty_json()
        .context("Could not serialize short-audio receipt JSON")?;
    if let Some(parent) = options.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Could not create receipt output directory {}",
                parent.display()
            )
        })?;
    }
    atomic_write_text(options.out, &format!("{json}\n")).with_context(|| {
        format!(
            "Could not write short-audio receipt to {}",
            options.out.display()
        )
    })?;
    eprintln!(
        "Wrote {} short-audio receipt to {}",
        SHORT_AUDIO_RECEIPT_SCHEMA,
        options.out.display()
    );
    Ok(())
}

#[derive(Debug, Clone)]
struct PackBinding {
    model_id: String,
    content_sha256: String,
    size_bytes: u64,
    quant: String,
}

fn resolve_pack_binding(
    model_arg: Option<&str>,
    resolved_model_id: &str,
    model_pack_path: Option<&Path>,
    backend_kind: BackendKind,
    home: &Path,
) -> Result<PackBinding> {
    let packs = list_installed_packs(home).unwrap_or_default();

    if let Some(path) = model_pack_path {
        let validated = validate_local_native_model_pack_path(path)
            .map_err(|error| anyhow::anyhow!("Native model-pack path rejected: {error}"))?;
        let (size_bytes, content_sha256) = sha256_file(&validated)
            .with_context(|| format!("Could not hash model pack {}", validated.display()))?;
        let quant = packs
            .iter()
            .find(|pack| pack.path == validated)
            .map(|pack| pack.quant.clone())
            .or_else(|| quant_from_model_ref(model_arg))
            .unwrap_or_else(|| "unknown".to_string());
        let model_id = display_model_id(model_arg, resolved_model_id, &quant);
        return Ok(PackBinding {
            model_id,
            content_sha256,
            size_bytes,
            quant,
        });
    }

    if let Some(model_arg) = model_arg
        && let Some(pack) = find_installed_pack(&packs, model_arg)
    {
        return Ok(binding_from_installed(model_arg, pack));
    }

    if let Some(pack) = packs.iter().find(|pack| {
        pack.model_id == resolved_model_id
            || pack.pull == resolved_model_id
            || pack.filename == resolved_model_id
    }) {
        return Ok(binding_from_installed(
            model_arg.unwrap_or(resolved_model_id),
            pack,
        ));
    }

    if backend_kind == BackendKind::Mock {
        // Mock has no pack bytes; bind a stable placeholder digest so the
        // schema stays complete without claiming pack verification.
        let quant = quant_from_model_ref(model_arg).unwrap_or_else(|| "unknown".to_string());
        return Ok(PackBinding {
            model_id: display_model_id(model_arg, resolved_model_id, &quant),
            content_sha256: "0".repeat(64),
            size_bytes: 0,
            quant,
        });
    }

    bail!(
        "Could not bind pack content for model '{}'. Install it or pass --model-pack.",
        model_arg.unwrap_or(resolved_model_id)
    )
}

fn binding_from_installed(model_arg: &str, pack: &InstalledPack) -> PackBinding {
    PackBinding {
        model_id: display_model_id(Some(model_arg), &pack.model_id, &pack.quant),
        content_sha256: pack.sha256.clone(),
        size_bytes: pack.size_bytes,
        quant: pack.quant.clone(),
    }
}

fn find_installed_pack<'a>(
    packs: &'a [InstalledPack],
    model_arg: &str,
) -> Option<&'a InstalledPack> {
    if let Ok(Some(pack)) = resolve_installed_pack_reference(packs, model_arg) {
        let path = pack.path;
        return packs.iter().find(|candidate| candidate.path == path);
    }
    let parsed = parse_model_ref(model_arg).ok();
    packs.iter().find(|pack| {
        pack.pull == model_arg
            || pack.model_id == model_arg
            || parsed.as_ref().is_some_and(|model_ref| {
                pack.model_id == model_ref.family
                    && model_ref.tag.as_deref().is_none_or(|tag| {
                        openasr_core::canonical_quant_tag(tag)
                            == openasr_core::canonical_quant_tag(&pack.quant)
                            || openasr_core::canonical_quant_tag(tag)
                                == openasr_core::canonical_quant_tag(&pack.suffix)
                    })
            })
    })
}

fn display_model_id(model_arg: Option<&str>, resolved_model_id: &str, quant: &str) -> String {
    if let Some(model_arg) = model_arg.map(str::trim).filter(|value| !value.is_empty()) {
        if model_arg.contains(':') {
            return model_arg.to_string();
        }
        if quant != "unknown" {
            return format!("{model_arg}:{quant}");
        }
        return model_arg.to_string();
    }
    if quant != "unknown" {
        format!("{resolved_model_id}:{quant}")
    } else {
        resolved_model_id.to_string()
    }
}

fn quant_from_model_ref(model_arg: Option<&str>) -> Option<String> {
    let model_arg = model_arg?;
    let parsed = parse_model_ref(model_arg).ok()?;
    parsed
        .tag
        .map(|tag| openasr_core::canonical_quant_tag(&tag).to_string())
}

fn parse_receipt_device(raw: &str) -> Result<ExecutionTarget> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" => Ok(ExecutionTarget::Cpu),
        "auto" => Ok(ExecutionTarget::Auto),
        "accelerated" | "metal" | "cuda" | "hip" | "vulkan" | "gpu" => {
            Ok(ExecutionTarget::Accelerated)
        }
        other => bail!(
            "Unsupported --device '{other}'. Use one of: cpu, metal, cuda, accelerated, auto."
        ),
    }
}

fn normalize_device_label(raw: &str) -> String {
    raw.trim().to_ascii_lowercase()
}

fn build_command_argv(
    options: &ShortAudioReceiptOptions<'_>,
    pack: &PackBinding,
    device_label: &str,
) -> Vec<String> {
    let mut command = vec![
        "openasr".to_string(),
        "bench-receipt".to_string(),
        "short-audio".to_string(),
        "--audio".to_string(),
        options.audio.display().to_string(),
        "--backend".to_string(),
        options.backend_kind.to_string(),
        "--device".to_string(),
        device_label.to_string(),
        "--out".to_string(),
        options.out.display().to_string(),
        "--runs".to_string(),
        options.runs.to_string(),
        "--warmup-runs".to_string(),
        options.warmup_runs.to_string(),
        "--scope".to_string(),
        options.scope.to_string(),
    ];
    if let Some(model) = options.model {
        command.push("--model".to_string());
        command.push(model.to_string());
    } else {
        command.push("--model".to_string());
        command.push(pack.model_id.clone());
    }
    if let Some(model_pack) = options.model_pack {
        command.push("--model-pack".to_string());
        command.push(model_pack.display().to_string());
    }
    if let Some(core_commit) = options.core_commit {
        command.push("--core-commit".to_string());
        command.push(core_commit.to_string());
    }
    command
}

fn capture_env_allowlist() -> BTreeMap<String, String> {
    const KEYS: &[&str] = &[
        "OPENASR_HOME",
        "OPENASR_GGML_BACKEND",
        "OPENASR_BUILD_COMMIT",
        "OPENASR_OFFLINE",
    ];
    let mut out = BTreeMap::new();
    for key in KEYS {
        if let Ok(value) = std::env::var(key)
            && !value.is_empty()
        {
            out.insert((*key).to_string(), value);
        }
    }
    out
}

fn validate_observed_accelerator_placement(
    requested_device: &str,
    observed: &GgmlExecutionPlacementSummary,
) -> Result<()> {
    if !requested_device.eq_ignore_ascii_case("metal") {
        return Ok(());
    }
    let mut metal_nodes = 0_u64;
    let mut non_metal = BTreeMap::<String, u64>::new();
    for (backend, nodes) in &observed.observed_compute_nodes_by_backend {
        let normalized = backend.to_ascii_lowercase();
        if normalized.contains("metal") || normalized.starts_with("mtl") {
            metal_nodes = metal_nodes.saturating_add(*nodes);
        } else if *nodes > 0 {
            non_metal.insert(backend.clone(), *nodes);
        }
    }
    if metal_nodes == 0 {
        bail!(
            "Metal receipt observed no Metal compute nodes; observed={:?}",
            observed.observed_compute_nodes_by_backend
        );
    }
    if !non_metal.is_empty() {
        bail!("Metal receipt observed compute-node fallback outside Metal: {non_metal:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openasr_core::{SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE, SHORT_AUDIO_RECEIPT_SCHEMA};
    use tempfile::TempDir;

    #[test]
    fn parse_receipt_device_maps_accelerators() {
        assert_eq!(parse_receipt_device("cpu").unwrap(), ExecutionTarget::Cpu);
        assert_eq!(
            parse_receipt_device("metal").unwrap(),
            ExecutionTarget::Accelerated
        );
        assert_eq!(
            parse_receipt_device("CUDA").unwrap(),
            ExecutionTarget::Accelerated
        );
        assert!(parse_receipt_device("tpu").is_err());
    }

    #[test]
    fn mock_short_audio_receipt_roundtrip_on_fixture_audio() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
        if !fixture.is_file() {
            eprintln!("skip: fixtures/jfk.wav not found at {}", fixture.display());
            return;
        }

        let dir = TempDir::new().unwrap();
        let home = dir.path().join("home");
        std::fs::create_dir_all(&home).unwrap();
        let out = dir.path().join("receipt.json");

        let services = Arc::new(
            NativeExecutionServices::for_local_process()
                .expect("test execution services must construct"),
        );
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let previous_home = std::env::var_os("OPENASR_HOME");
        // SAFETY: test-only process env mutation; nextest isolates processes.
        unsafe {
            std::env::set_var("OPENASR_HOME", &home);
        }
        let result = bench_receipt_short_audio(
            &services,
            ShortAudioReceiptOptions {
                model: Some("whisper-tiny"),
                audio: &fixture,
                backend_kind: BackendKind::Mock,
                device: "cpu",
                model_pack: None,
                out: &out,
                runs: 1,
                warmup_runs: 0,
                core_commit: Some(commit),
                scope: SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE,
                ffmpeg_bin: None,
                git_cwd: None,
            },
        );
        match previous_home {
            Some(value) => unsafe {
                std::env::set_var("OPENASR_HOME", value);
            },
            None => unsafe {
                std::env::remove_var("OPENASR_HOME");
            },
        }
        result.expect("mock short-audio receipt should succeed");

        let raw = std::fs::read_to_string(&out).unwrap();
        let receipt = ShortAudioReceipt::from_json_str(&raw).unwrap();
        assert_eq!(receipt.schema, SHORT_AUDIO_RECEIPT_SCHEMA);
        assert_eq!(receipt.core_commit, commit);
        assert_eq!(receipt.run.backend, "mock");
        assert_eq!(receipt.run.device, "cpu");
        assert_eq!(receipt.placement, "cpu");
        assert!(!receipt.transcript.text.is_empty());
        assert_eq!(receipt.audio.sha256.len(), 64);
        assert!(receipt.metrics.rtf_samples.len() <= 1);
    }

    #[test]
    fn display_model_id_prefers_explicit_quant_ref() {
        assert_eq!(
            display_model_id(Some("funasr-nano:q4"), "funasr-nano", "q4_k"),
            "funasr-nano:q4"
        );
        assert_eq!(
            display_model_id(Some("funasr-nano"), "funasr-nano", "q4_k"),
            "funasr-nano:q4_k"
        );
    }

    #[test]
    fn metal_placement_gate_ignores_metadata_views_but_rejects_compute_fallback() {
        let mut observed = GgmlExecutionPlacementSummary {
            direct_graph_computes: 0,
            scheduler_graph_computes: 1,
            observed_nodes_by_backend: BTreeMap::from([
                ("CPU".to_string(), 1),
                ("MTL0".to_string(), 20),
            ]),
            observed_compute_nodes_by_backend: BTreeMap::from([("MTL0".to_string(), 19)]),
            observed_node_output_bytes_by_backend: BTreeMap::new(),
            fallback_node_samples_by_backend: BTreeMap::new(),
        };
        validate_observed_accelerator_placement("metal", &observed).unwrap();
        observed
            .observed_compute_nodes_by_backend
            .insert("CPU".to_string(), 1);
        assert!(validate_observed_accelerator_placement("metal", &observed).is_err());
    }
}
