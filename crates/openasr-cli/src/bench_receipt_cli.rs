//! `openasr bench-receipt short-audio`  -  emit a machine-readable short-audio
//! audit receipt (`openasr.short-audio-receipt.v0`).
//!
//! Explicit tooling command: not part of the default `transcribe` user path.
//! Reuses the same transcription ingress as `transcribe --benchmark`.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
#[cfg(test)]
use std::{fs, io::Write};

use anyhow::{Context, Result, bail};
use openasr_core::{
    BackendKind, ExecutionTarget, GgmlExecutionPlacementSummary, GgmlExecutionTelemetryCollector,
    InstalledPack, NativeExecutionReceiptCollector, NativeExecutionReceiptSnapshot,
    NativeExecutionServices, RequestAttemptId, RequestExecutionTerminal, ResolvedOutputTarget,
    SHORT_AUDIO_RECEIPT_MEASUREMENT_WALL_CLOCK, SHORT_AUDIO_RECEIPT_SCHEMA,
    ShortAudioExecutionProjection, ShortAudioReceipt, ShortAudioReceiptAudio,
    ShortAudioReceiptDecodeDiagnostics, ShortAudioReceiptMetrics, ShortAudioReceiptPack,
    ShortAudioReceiptRun, ShortAudioReceiptTranscript, TranscriptionRequest, atomic_write_text,
    ggml_runtime::{
        AutoGpuPolicy, GgmlDecodeLogitsConsumers, GgmlDecodeOutputContract,
        RequestBackendPreference, ResolvedFamilyRuntimeInput,
    },
    list_installed_packs, load_config, openasr_home, parse_model_ref, prepare_audio_input,
    process_memory_snapshot, receipt_os_id, resolve_core_commit, resolve_installed_pack_reference,
    resolve_output_target_handle, sha256_file, validate_local_native_model_pack_path,
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
    pub(crate) trace_out: Option<&'a Path>,
}

/// Delegate release eligibility to the core-owned receipt predicate. The GPU
/// matrix gate remains the approval authority; this command deliberately has
/// no matrix, catalog, target, or activation policy of its own.
pub(crate) fn validate_qualification_receipts(paths: &[PathBuf]) -> Result<()> {
    if paths.is_empty() {
        bail!("at least one --receipt is required");
    }
    for (index, path) in paths.iter().enumerate() {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("Could not read qualification receipt #{}", index + 1))?;
        let receipt = ShortAudioReceipt::from_json_str(&raw)
            .with_context(|| format!("Qualification receipt #{} is invalid", index + 1))?;
        receipt
            .validate_qualification_eligibility()
            .with_context(|| format!("Qualification receipt #{} is ineligible", index + 1))?;
    }
    println!("validated {} qualification receipt(s)", paths.len());
    Ok(())
}

pub(crate) fn bench_receipt_short_audio(
    native_execution_services: &Arc<NativeExecutionServices>,
    options: ShortAudioReceiptOptions<'_>,
) -> Result<()> {
    if options.runs == 0 {
        bail!("--runs must be >= 1");
    }
    if !privacy_safe_scope_label(options.scope) {
        bail!(
            "--scope must be a privacy-safe semantic label, optionally followed by one '/<32-lower-hex-nonce>' runner suffix"
        );
    }
    if !options.audio.is_file() {
        bail!(
            "Audio file not found: {}\nPass an existing WAV/audio path via --audio.",
            options.audio.display()
        );
    }
    if options.trace_out.is_some() && options.backend_kind != BackendKind::Native {
        bail!("--trace-out requires --backend native; mock cannot produce a native trace");
    }
    if options.trace_out.is_some()
        && let Some(parent) = options.out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Could not create receipt output directory {}",
                parent.display()
            )
        })?;
    }
    let fixed_output_targets = options
        .trace_out
        .map(|trace_out| fixed_receipt_and_trace_targets(options.out, trace_out))
        .transpose()?;
    if let Some((_, trace_target)) = &fixed_output_targets
        && trace_target.path().exists()
    {
        bail!("refusing to overwrite runtime trace output");
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
    // Native receipts must project decode_diagnostics from the runtime that
    // actually ran. Reconstructing output_plan/reuse_mode from --device or
    // --warmup-runs is forbidden.
    let mut last_request_receipt = None;
    let mut rtf_samples = Vec::with_capacity(options.runs);
    let mut last_text = String::new();
    let mut last_truncated = Vec::new();
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
        let pass_receipt = if options.backend_kind == BackendKind::Native {
            let attempt_id = RequestAttemptId::generate()
                .map_err(|_| anyhow::anyhow!("could not allocate request attempt identity"))?;
            let receipt = NativeExecutionReceiptCollector::new();
            Some((attempt_id, receipt))
        } else {
            None
        };
        let mut request = TranscriptionRequest::new(
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
        if let Some((attempt_id, receipt)) = &pass_receipt {
            request = request.with_execution_context(Arc::new(
                openasr_core::RequestExecutionContext::uncancellable(
                    "short-audio receipt command has no cancel surface",
                )
                .with_request_attempt_id(*attempt_id)
                .with_native_execution_receipt(receipt.clone()),
            ));
        }

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
        if let Some((_, receipt)) = &pass_receipt {
            receipt.record_terminal(RequestExecutionTerminal::Succeeded);
            last_request_receipt = Some(receipt.clone());
        }
        if options.trace_out.is_some()
            && transcription
                .longform
                .as_ref()
                .is_some_and(|longform| longform.chunk_count > 1)
        {
            bail!(
                "--trace-out rejects multi-slice native requests; trace schema has no slice identity"
            );
        }
        last_text = transcription.text;
        last_truncated = transcription.truncated_decodes;

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

    let audio_label = receipt_audio_label(&audio_sha256);
    let command = build_command_argv(&options, &pack_binding, &device_label, &audio_label);
    let env_allowlist = capture_env_allowlist(&core_commit);

    let memory_after_model = process_memory_snapshot();
    let observed_placement = execution_telemetry.snapshot();
    if options.backend_kind == BackendKind::Native {
        validate_observed_accelerator_placement(&device_label, &observed_placement)?;
    }
    let native_snapshot = last_request_receipt
        .as_ref()
        .map(NativeExecutionReceiptCollector::snapshot);
    let decode_diagnostics = project_decode_diagnostics(
        match options.backend_kind {
            BackendKind::Native => None,
            BackendKind::Mock => Some(resolved_runtime_for_mock_receipt(options.device)?),
        },
        native_snapshot.as_ref(),
    )?;
    if !last_truncated.is_empty() {
        for truncated in &last_truncated {
            notes.push(format!(
                "decode_stop={}",
                truncated.truncation.reason.as_str()
            ));
        }
    } else if native_snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.token_steps.last())
        .is_some_and(|step| step.is_eot)
    {
        notes.push("decode_stop=stop_token".to_string());
    }
    if let (Some((_, trace_target)), Some(snapshot)) =
        (fixed_output_targets.as_ref(), native_snapshot.as_ref())
    {
        let facts = snapshot.facts.as_ref().ok_or_else(|| {
            anyhow::anyhow!("native trace is missing request-scoped execution facts")
        })?;
        if !snapshot.completed
            || snapshot.trace.overflowed
            || snapshot.trace.event_count == 0
            || snapshot.trace.jsonl.is_empty()
            || facts.actual_provider != Some(facts.selected_provider)
            || facts.actual_stable_device_id.as_deref() != Some(facts.stable_device_id.as_str())
            || facts.scheduler_enabled.is_none()
        {
            bail!("native trace is incomplete; refusing to emit an approval trace");
        }
        validate_release_trace(&snapshot.trace.jsonl)?;
        trace_target
            .create_new_text_and_sync_parent(&snapshot.trace.jsonl)
            .with_context(|| {
                format!(
                    "Could not write runtime trace to {}",
                    trace_target.path().display()
                )
            })?;
    }
    // A generic native receipt records only typed placement telemetry produced
    // by the runtime. It is not a release correctness approval; token-transcript
    // evidence requires the runtime token-trace producer and immutable matrix
    // bindings. Strict trace publication above remains the only
    // approval-producing path.
    let receipt_evidence = None;
    let execution = native_snapshot.as_ref().map(|request_snapshot| {
        let runtime_snapshot = native_execution_services.runtime_receipts().snapshot();
        let reconciliation = native_execution_services
            .runtime_receipts()
            .reconcile_live_leases_quiescent(native_execution_services.memory_broker());
        ShortAudioExecutionProjection::from_receipts(
            request_snapshot,
            &runtime_snapshot,
            &reconciliation,
        )
    });
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
            path_or_label: audio_label,
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
        evidence: receipt_evidence,
        execution,
        scope: options.scope.to_string(),
        notes,
        decode_diagnostics: Some(decode_diagnostics),
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
    if let Some((receipt_target, _)) = fixed_output_targets.as_ref() {
        receipt_target
            .atomic_write_text(&format!("{json}\n"))
            .with_context(|| {
                format!(
                    "Could not write short-audio receipt to fixed target {}",
                    receipt_target.path().display()
                )
            })?;
    } else {
        atomic_write_text(options.out, &format!("{json}\n")).with_context(|| {
            format!(
                "Could not write short-audio receipt to {}",
                options.out.display()
            )
        })?;
    }
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
    if let Some(model_arg) = model_arg
        .map(str::trim)
        .filter(|value| !value.is_empty() && !looks_like_local_path(value))
    {
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

/// A receipt identifies a pack through its content digest and model identity;
/// caller-provided path spellings are neither stable nor safe to retain.
fn looks_like_local_path(value: &str) -> bool {
    value.contains('/') || value.contains('\\')
}

fn quant_from_model_ref(model_arg: Option<&str>) -> Option<String> {
    let model_arg = model_arg?;
    let parsed = parse_model_ref(model_arg).ok()?;
    parsed
        .tag
        .map(|tag| openasr_core::canonical_quant_tag(&tag).to_string())
}

fn project_decode_diagnostics(
    resolved: Option<ResolvedFamilyRuntimeInput>,
    snapshot: Option<&NativeExecutionReceiptSnapshot>,
) -> Result<ShortAudioReceiptDecodeDiagnostics> {
    openasr_core::decode_diagnostics_from_shipped_runtime(resolved.as_ref(), snapshot).map_err(
        |error| match error {
            openasr_core::ShortAudioReceiptError::DecodeDiagnosticsMissing => anyhow::anyhow!(
                "short-audio receipt is missing shipped output_plan/reuse_mode; refusing to emit a receipt"
            ),
            other => anyhow::anyhow!("{other}"),
        },
    )
}

/// Mock transcription never records native execution facts. Plumbing receipts
/// still project through the same planner, with no logits consumers.
fn resolved_runtime_for_mock_receipt(device: &str) -> Result<ResolvedFamilyRuntimeInput> {
    let preference = match parse_receipt_device(device)? {
        ExecutionTarget::Cpu => Some(RequestBackendPreference::CpuOnly),
        ExecutionTarget::Accelerated => Some(RequestBackendPreference::Accelerated),
        ExecutionTarget::Auto => None,
    };
    Ok(
        ResolvedFamilyRuntimeInput::resolve_with_output_contract_and_consumers(
            preference,
            AutoGpuPolicy::AllBackends,
            GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
            GgmlDecodeLogitsConsumers::none(),
        ),
    )
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

fn privacy_safe_scope_label(value: &str) -> bool {
    if privacy_safe_scope_segment(value) {
        return true;
    }
    let mut segments = value.split('/');
    let Some(base) = segments.next() else {
        return false;
    };
    let nonce = segments.next();
    segments.next().is_none()
        && privacy_safe_scope_segment(base)
        && nonce.is_some_and(|nonce| {
            nonce.len() == 32
                && nonce
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
}

fn privacy_safe_scope_segment(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 256
        && bytes[0].is_ascii_alphanumeric()
        && bytes.iter().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'-' | b'_' | b'.' | b':' | b'+' | b'@' | b'=')
        })
        && !(bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
}

/// Returns a stable, byte-bound fixture label safe to retain in a receipt.
/// Local file paths are ingress-only and must never become evidence payload.
fn receipt_audio_label(audio_sha256: &str) -> String {
    format!("audio-sha256:{audio_sha256}")
}

fn build_command_argv(
    options: &ShortAudioReceiptOptions<'_>,
    pack: &PackBinding,
    device_label: &str,
    audio_label: &str,
) -> Vec<String> {
    let mut command = vec![
        "openasr".to_string(),
        "bench-receipt".to_string(),
        "short-audio".to_string(),
        "--audio".to_string(),
        audio_label.to_string(),
        "--backend".to_string(),
        options.backend_kind.to_string(),
        "--device".to_string(),
        device_label.to_string(),
        "--out".to_string(),
        "receipt-output".to_string(),
        "--runs".to_string(),
        options.runs.to_string(),
        "--warmup-runs".to_string(),
        options.warmup_runs.to_string(),
        "--scope".to_string(),
        options.scope.to_string(),
    ];
    command.push("--model".to_string());
    command.push(pack.model_id.clone());
    if options.model_pack.is_some() {
        command.push("--model-pack".to_string());
        command.push(format!("pack-content-sha256:{}", pack.content_sha256));
    }
    if let Some(core_commit) = options.core_commit {
        command.push("--core-commit".to_string());
        command.push(core_commit.to_string());
    }
    if options.trace_out.is_some() {
        command.push("--trace-out".to_string());
        command.push("runtime-trace-output".to_string());
    }
    command
}

fn capture_env_allowlist(core_commit: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Ok(value) = std::env::var("OPENASR_GGML_BACKEND") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(
            normalized.as_str(),
            "cpu" | "metal" | "gpu" | "cuda" | "hip" | "vulkan"
        ) {
            out.insert("OPENASR_GGML_BACKEND".to_string(), normalized);
        }
    }
    if let Ok(value) = std::env::var("OPENASR_BUILD_COMMIT") {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized == core_commit && openasr_core::validate_core_commit(&normalized).is_ok() {
            out.insert("OPENASR_BUILD_COMMIT".to_string(), normalized);
        }
    }
    if std::env::var("OPENASR_OFFLINE").ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    }) {
        out.insert("OPENASR_OFFLINE".to_string(), "true".to_string());
    }
    out
}

#[cfg(test)]
fn resolved_output_target(path: &Path) -> Result<PathBuf> {
    openasr_core::resolve_output_target(path)
        .with_context(|| format!("Could not resolve output target {}", path.display()))
}

fn fixed_receipt_and_trace_targets(
    receipt: &Path,
    trace: &Path,
) -> Result<(ResolvedOutputTarget, ResolvedOutputTarget)> {
    let receipt = resolve_output_target_handle(receipt)?;
    let trace = resolve_output_target_handle(trace)?;
    if receipt.path() == trace.path() {
        bail!("--out and --trace-out must name different output targets");
    }
    Ok((receipt, trace))
}

/// Write a complete trace through a same-directory temporary file, then publish
/// it with a hard link. `hard_link` is create-new: an attacker or concurrent
/// producer that creates the destination after validation wins safely and we
/// fail rather than replacing their file.
#[cfg(test)]
fn atomic_create_new_trace_at_target(target: &Path, contents: &str) -> Result<()> {
    atomic_create_new_trace_at_target_with(target, contents, |parent| {
        fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(anyhow::Error::from)
    })
}

#[cfg(test)]
fn atomic_create_new_trace_at_target_with(
    target: &Path,
    contents: &str,
    sync_parent: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let parent = target.parent().expect("fixed target has a parent");
    let mut temp = tempfile::NamedTempFile::new_in(parent).with_context(|| {
        format!(
            "Could not create trace temporary file in {}",
            parent.display()
        )
    })?;
    temp.write_all(contents.as_bytes())
        .context("Could not write runtime trace temporary file")?;
    temp.as_file()
        .sync_all()
        .context("Could not sync runtime trace temporary file")?;
    if let Err(error) = fs::hard_link(temp.path(), target) {
        let _ = temp.close();
        return Err(anyhow::anyhow!(
            "Could not create runtime trace {} without replacing an existing file: {error}",
            target.display()
        ));
    }
    if let Err(error) = sync_parent(parent) {
        // The create-new artifact remains intact: unlinking by path after a
        // metadata failure could race a replacement and delete another writer's
        // file. It is not release-valid because this call fails closed.
        let _ = temp.close();
        return Err(anyhow::anyhow!(
            "Could not persist runtime trace directory metadata for {}: {error}; artifact retained and unusable",
            target.display(),
        ));
    }
    temp.close()
        .context("Could not remove runtime trace temporary file")?;
    Ok(())
}

fn validate_release_trace(jsonl: &str) -> Result<()> {
    let mut token_steps = BTreeSet::new();
    let mut top_k_steps = BTreeSet::new();
    let mut logits_steps = BTreeSet::new();
    for line in jsonl.lines() {
        let event: serde_json::Value =
            serde_json::from_str(line).context("runtime trace contains invalid JSON")?;
        let step = event.get("step_index").and_then(serde_json::Value::as_u64);
        match event.get("event").and_then(serde_json::Value::as_str) {
            Some("token") => {
                let step =
                    step.ok_or_else(|| anyhow::anyhow!("runtime token trace has no step_index"))?;
                if !token_steps.insert(step) {
                    bail!("runtime token trace has duplicate step_index {step}");
                }
            }
            Some("top_k") => {
                let step =
                    step.ok_or_else(|| anyhow::anyhow!("runtime top-k trace has no step_index"))?;
                let items = event
                    .get("items")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| anyhow::anyhow!("runtime top-k trace has no items"))?;
                if items.len() < 2
                    || event
                        .get("top1_top2_margin")
                        .and_then(serde_json::Value::as_f64)
                        .is_none()
                {
                    bail!("runtime top-k trace lacks a real top1/top2 margin at step {step}");
                }
                if !top_k_steps.insert(step) {
                    bail!("runtime top-k trace has duplicate step_index {step}");
                }
            }
            Some("logits_digest") => {
                let step =
                    step.ok_or_else(|| anyhow::anyhow!("runtime logits digest has no step_index"))?;
                let digest = event.get("sha256").and_then(serde_json::Value::as_str);
                if event
                    .get("element_count")
                    .and_then(serde_json::Value::as_u64)
                    .is_none_or(|count| count == 0)
                    || event
                        .get("non_finite_count")
                        .and_then(serde_json::Value::as_u64)
                        != Some(0)
                    || digest.is_none_or(|value| {
                        value.len() != 64
                            || value.bytes().any(|byte| {
                                !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte)
                            })
                    })
                {
                    bail!("runtime logits digest is invalid at step {step}");
                }
                if !logits_steps.insert(step) {
                    bail!("runtime logits trace has duplicate step_index {step}");
                }
            }
            _ => {}
        }
    }
    if token_steps.is_empty() || token_steps != top_k_steps || token_steps != logits_steps {
        bail!(
            "every runtime token trace step must have exactly one same-step top-k/margin and logits-digest record"
        );
    }
    Ok(())
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
    use openasr_core::{
        SHORT_AUDIO_RECEIPT_DEFAULT_SCOPE, SHORT_AUDIO_RECEIPT_SCHEMA, ShortAudioReceiptOutputPlan,
        ShortAudioReceiptReuseMode,
    };
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
    fn receipt_scope_accepts_the_runner_nonce_without_accepting_paths() {
        assert!(privacy_safe_scope_label(&format!(
            "hardware-evidence/{}",
            "a".repeat(32)
        )));
        for scope in [
            "/home/alice",
            "C:\\Users\\alice",
            "\\\\server\\share",
            "../0123456789abcdef0123456789abcdef",
            "scope/not-a-nonce",
        ] {
            assert!(!privacy_safe_scope_label(scope), "{scope}");
        }
    }

    fn exact_accelerated_preference(
        provider: openasr_core::ExecutionProvider,
    ) -> RequestBackendPreference {
        RequestBackendPreference::Exact(openasr_core::ResolvedExecutionRoute {
            provider,
            stable_id: format!("{}0", provider.as_str()),
            registry_ordinal: 0,
            kind: openasr_core::RouteDeviceKind::Accelerated,
            addressability: openasr_core::DeviceAddressability::NotExactlyAddressable {
                reason: "receipt output-plan fixture",
            },
        })
    }

    fn resolved_runtime(
        preference: Option<RequestBackendPreference>,
        consumers: GgmlDecodeLogitsConsumers,
    ) -> ResolvedFamilyRuntimeInput {
        ResolvedFamilyRuntimeInput::resolve_with_output_contract_and_consumers(
            preference,
            AutoGpuPolicy::AllBackends,
            GgmlDecodeOutputContract::NativeFirstMaxTokenOrFullLogits,
            consumers,
        )
    }

    #[test]
    fn shipped_emitter_projects_cpu_native_first_max_without_logits_consumers() {
        let resolved = resolved_runtime(
            Some(RequestBackendPreference::CpuOnly),
            GgmlDecodeLogitsConsumers::none(),
        );
        let diagnostics = project_decode_diagnostics(Some(resolved), None)
            .expect("CPU compact runtime must project");
        assert_eq!(
            diagnostics.output_plan,
            ShortAudioReceiptOutputPlan::from(resolved.output_plan())
        );
        assert_eq!(
            diagnostics.reuse_mode,
            ShortAudioReceiptReuseMode::from(resolved.reuse_mode())
        );
        assert_eq!(
            diagnostics.output_plan,
            ShortAudioReceiptOutputPlan::NativeFirstMaxToken
        );
        assert_eq!(
            diagnostics.reuse_mode,
            ShortAudioReceiptReuseMode::FreshGraph
        );
    }

    #[test]
    fn shipped_emitter_projects_unproven_metal_and_gpu_full_logits() {
        for provider in [
            openasr_core::ExecutionProvider::Metal,
            openasr_core::ExecutionProvider::Cuda,
            openasr_core::ExecutionProvider::Vulkan,
            openasr_core::ExecutionProvider::Hip,
        ] {
            let resolved = resolved_runtime(
                Some(exact_accelerated_preference(provider)),
                GgmlDecodeLogitsConsumers::none(),
            );
            let diagnostics = project_decode_diagnostics(Some(resolved), None)
                .expect("unproven accelerator runtime must project");
            assert_eq!(
                diagnostics.output_plan,
                ShortAudioReceiptOutputPlan::from(resolved.output_plan())
            );
            assert_eq!(
                diagnostics.output_plan,
                ShortAudioReceiptOutputPlan::FullLogits,
                "unproven {provider:?} must not claim compact output"
            );
            assert_eq!(
                diagnostics.reuse_mode,
                ShortAudioReceiptReuseMode::FreshGraph
            );
        }
    }

    #[test]
    fn shipped_emitter_cpu_logits_consumers_force_full_logits() {
        let resolved = resolved_runtime(
            Some(RequestBackendPreference::CpuOnly),
            GgmlDecodeLogitsConsumers::none().with_phrase_bias(true),
        );
        let diagnostics = project_decode_diagnostics(Some(resolved), None)
            .expect("CPU complete-logits runtime must project");
        assert_eq!(
            diagnostics.output_plan,
            ShortAudioReceiptOutputPlan::FullLogits
        );
    }

    #[test]
    fn shipped_emitter_fail_closed_without_resolved_runtime() {
        let error = project_decode_diagnostics(None, None)
            .expect_err("missing shipped plan/reuse must not emit");
        assert!(error.to_string().contains("output_plan/reuse_mode"));
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
                trace_out: None,
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
        let diagnostics = receipt
            .decode_diagnostics
            .as_ref()
            .expect("decode diagnostics are required");
        let expected = project_decode_diagnostics(
            Some(resolved_runtime_for_mock_receipt("cpu").expect("cpu mock runtime")),
            None,
        )
        .expect("mock CPU path must project a shipped runtime");
        assert_eq!(diagnostics.output_plan, expected.output_plan);
        assert_eq!(diagnostics.reuse_mode, expected.reuse_mode);
        assert_eq!(
            diagnostics.output_plan,
            ShortAudioReceiptOutputPlan::NativeFirstMaxToken
        );
        assert_eq!(
            diagnostics.reuse_mode,
            ShortAudioReceiptReuseMode::FreshGraph
        );
        assert!(!receipt.transcript.text.is_empty());
        assert_eq!(receipt.audio.sha256.len(), 64);
        assert_eq!(
            receipt.audio.path_or_label,
            receipt_audio_label(&receipt.audio.sha256)
        );
        assert!(
            !raw.contains(fixture.to_string_lossy().as_ref()),
            "receipt must not retain the caller audio path"
        );
        assert!(
            !raw.contains(home.to_string_lossy().as_ref()),
            "receipt must not retain OPENASR_HOME"
        );
        assert!(
            !raw.contains(out.to_string_lossy().as_ref()),
            "receipt command must not retain its output path"
        );
        assert!(
            !receipt
                .run
                .command
                .iter()
                .any(|part| part.contains('/') || part.contains('\\')),
            "receipt command must contain only privacy-safe argument labels"
        );
        assert!(receipt.metrics.rtf_samples.len() <= 1);
    }

    #[test]
    fn receipt_command_replaces_caller_paths_with_stable_bindings() {
        let audio = PathBuf::from("/private/var/folders/example/alice/recording.wav");
        let out = PathBuf::from("/home/alice/receipt.json");
        let model_pack = PathBuf::from(r"C:\Users\alice\AppData\Local\model.oasr");
        let trace_out = PathBuf::from("/tmp/openasr/trace.jsonl");
        let options = ShortAudioReceiptOptions {
            model: Some(r"C:\Users\alice\AppData\Local\model.oasr"),
            audio: &audio,
            backend_kind: BackendKind::Native,
            device: "cuda",
            model_pack: Some(&model_pack),
            out: &out,
            runs: 1,
            warmup_runs: 0,
            core_commit: Some("0123456789abcdef0123456789abcdef01234567"),
            scope: "fixture",
            ffmpeg_bin: None,
            git_cwd: None,
            trace_out: Some(&trace_out),
        };
        let pack = PackBinding {
            model_id: "whisper-tiny:q4_k".to_string(),
            content_sha256: "a".repeat(64),
            size_bytes: 1,
            quant: "q4_k".to_string(),
        };
        let audio_label = receipt_audio_label(&"b".repeat(64));
        let command = build_command_argv(&options, &pack, "cuda", &audio_label);
        let command_text = command.join("\u{0}");

        assert!(command.contains(&audio_label));
        assert!(command.contains(&"receipt-output".to_string()));
        assert!(command.contains(&"runtime-trace-output".to_string()));
        for forbidden in [
            "/private/var",
            "/home/alice",
            r"C:\Users\alice",
            "/tmp/openasr",
        ] {
            assert!(
                !command_text.contains(forbidden),
                "receipt command leaked caller path fragment {forbidden}"
            );
        }
        assert!(
            command
                .iter()
                .any(|part| { part == &format!("pack-content-sha256:{}", pack.content_sha256) })
        );
    }

    #[test]
    fn fixed_targets_collapse_lexical_trace_aliases() {
        let dir = TempDir::new().unwrap();
        let trace = dir.path().join("trace.jsonl");
        std::fs::write(&trace, "existing").unwrap();
        let dot_trace = dir.path().join(".").join("trace.jsonl");
        let parent_trace = dir.path().join("missing").join("..").join("trace.jsonl");
        assert!(fixed_receipt_and_trace_targets(&dot_trace, &trace).is_err());
        assert!(fixed_receipt_and_trace_targets(&parent_trace, &trace).is_err());
        assert_eq!(
            resolved_output_target(&trace).unwrap(),
            resolved_output_target(&dot_trace).unwrap()
        );
        assert_eq!(
            resolved_output_target(&trace).unwrap(),
            resolved_output_target(&parent_trace).unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn fixed_targets_reject_parent_directory_symlink_aliases() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real");
        let alias = dir.path().join("alias");
        std::fs::create_dir(&real).unwrap();
        symlink("real", &alias).unwrap();
        let via_alias = alias.join("trace.jsonl");
        let direct = real.join("trace.jsonl");
        assert_eq!(
            resolved_output_target(&via_alias).unwrap(),
            resolved_output_target(&direct).unwrap()
        );
        assert!(fixed_receipt_and_trace_targets(&via_alias, &direct).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn fixed_receipt_target_survives_post_resolution_parent_symlink_swap() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let real = dir.path().join("real");
        let other = dir.path().join("other");
        let alias = dir.path().join("alias");
        std::fs::create_dir(&real).unwrap();
        std::fs::create_dir(&other).unwrap();
        symlink("real", &alias).unwrap();
        let requested = alias.join("receipt.json");
        let fixed = resolve_output_target_handle(&requested).unwrap();
        std::fs::remove_file(&alias).unwrap();
        symlink("other", &alias).unwrap();
        fixed.atomic_write_text("receipt").unwrap();
        assert_eq!(
            std::fs::read_to_string(real.join("receipt.json")).unwrap(),
            "receipt"
        );
        assert!(!other.join("receipt.json").exists());
    }

    #[cfg(unix)]
    #[test]
    fn fixed_receipt_target_safely_replaces_post_resolution_symlink_swap() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let receipt = dir.path().join("receipt.json");
        let victim = dir.path().join("victim.json");
        std::fs::write(&victim, "victim").unwrap();
        let fixed = resolve_output_target_handle(&receipt).unwrap();
        symlink("victim.json", &receipt).unwrap();
        fixed.atomic_write_text("receipt").unwrap();
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "victim");
        assert_eq!(std::fs::read_to_string(&receipt).unwrap(), "receipt");
    }

    #[cfg(unix)]
    #[test]
    fn output_alias_rejects_dangling_receipt_symlink_to_trace() {
        use std::os::unix::fs::symlink;

        let dir = TempDir::new().unwrap();
        let receipt = dir.path().join("receipt.json");
        let trace = dir.path().join("trace.jsonl");
        symlink("trace.jsonl", &receipt).unwrap();
        // The link remains dangling until trace publication, but it already
        // resolves to the same final write target.
        assert!(fixed_receipt_and_trace_targets(&receipt, &trace).is_err());
        assert!(!trace.exists());
    }

    #[test]
    fn strict_trace_requires_same_step_logits_top_k_and_margin() {
        let valid = concat!(
            "{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"header\"}\n",
            "{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"token\",\"step_index\":0}\n",
            "{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"logits_digest\",\"step_index\":0,\"element_count\":2,\"sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"non_finite_count\":0}\n",
            "{\"schema\":\"openasr.gpu-correctness-trace.v1\",\"event\":\"top_k\",\"step_index\":0,\"items\":[{\"token_id\":1,\"value\":2.0},{\"token_id\":2,\"value\":1.0}],\"top1_top2_margin\":1.0}\n"
        );
        validate_release_trace(valid).expect("complete trace accepted");
        assert!(validate_release_trace("{\"event\":\"token\",\"step_index\":0}\n").is_err());
    }

    #[test]
    fn trace_create_new_refuses_racing_existing_target() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trace.jsonl");
        std::fs::write(&path, "other producer").unwrap();
        assert!(atomic_create_new_trace_at_target(&path, "trace\n").is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "other producer");
    }

    #[test]
    fn trace_directory_sync_failure_retains_unapproved_artifact() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("trace.jsonl");
        let error = atomic_create_new_trace_at_target_with(&path, "trace\n", |parent| {
            assert!(path.exists(), "link occurs before parent directory sync");
            assert_eq!(parent, dir.path());
            Err(anyhow::anyhow!("directory sync unavailable"))
        })
        .expect_err("directory sync failure must fail closed");
        assert!(error.to_string().contains("directory metadata"));
        assert!(
            path.exists(),
            "conservative failure retains create-new artifact"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "trace\n");
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
        assert_eq!(
            display_model_id(
                Some("/home/alice/.openasr/models/funasr-nano.oasr"),
                "funasr-nano",
                "q4_k"
            ),
            "funasr-nano:q4_k"
        );
        assert_eq!(
            display_model_id(
                Some(r"C:\Users\alice\AppData\Local\funasr-nano.oasr"),
                "funasr-nano",
                "q4_k"
            ),
            "funasr-nano:q4_k"
        );
    }

    #[test]
    fn shipped_emitter_projects_token_steps_from_collector_record_token() {
        let resolved = resolved_runtime(
            Some(RequestBackendPreference::CpuOnly),
            GgmlDecodeLogitsConsumers::none(),
        );
        let collector = NativeExecutionReceiptCollector::new();
        collector.record_top_k(0, &[2.0, 1.0]);
        collector.record_token(0, 11, false);
        collector.record_token(1, 7, true);
        let snapshot = collector.snapshot();
        let diagnostics = project_decode_diagnostics(Some(resolved), Some(&snapshot))
            .expect("collector token steps must project");
        assert_eq!(diagnostics.steps.len(), 2);
        assert_eq!(diagnostics.steps[0].step, 0);
        assert_eq!(diagnostics.steps[0].token_id, Some(11));
        assert_eq!(diagnostics.steps[0].top2_margin, Some(1.0));
        assert!(diagnostics.steps[0].graph_rebuilt);
        assert_eq!(diagnostics.steps[1].token_id, Some(7));
        assert_eq!(diagnostics.steps[1].top2_margin, None);
        assert!(snapshot.trace.event_count > 0);
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

    #[test]
    fn qualification_cli_delegates_to_strict_core_receipt_validation() {
        let temp = TempDir::new().unwrap();
        let receipt = temp.path().join("legacy.json");
        fs::write(&receipt, r#"{"schema":"openasr.short-audio-receipt.v0"}"#).unwrap();
        let error = validate_qualification_receipts(&[receipt]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Qualification receipt #1 is invalid")
        );
    }
}
