use std::{
    env, fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use openasr_core::api::backend::transcribe_with_mock_backend;
use openasr_core::{
    AudioInputInfo, AudioInputIssue, AudioPreparationOptions, BackendKind, BatchFailure,
    BatchOutput, BatchSummary, BenchmarkFormat, BenchmarkResult, CohereLocalSourceImportRequest,
    ConfigKey, DEFAULT_BACKEND_ID, DEFAULT_MODEL_ID, ModelCard, NATIVE_RUNTIME_MODEL_ID_AUTO,
    NativeBackend, OpenAsrConfig, PreparedAudioInput, Qwen3AsrLocalSourceImportRequest,
    ResponseFormat, TranscriptionBackend, TranscriptionRequest, WhisperLocalSourceImportRequest,
    atomic_write_text, config_path, convert_local_cohere_source_to_runtime_pack,
    convert_local_qwen_source_to_runtime_pack, convert_local_whisper_hf_source_to_runtime_pack,
    default_registry_dir, derive_catalog_public_key_hex, discover_batch_inputs, load_config,
    load_registry, openasr_home, parse_model_catalog, parse_model_ref, render_batch_summary,
    render_benchmark, render_catalog_signature_manifest, resolve_registry_model_ref,
    resolve_runtime_model_ref, save_config, validate_local_native_model_pack_path,
    verify_catalog_signature_manifest,
};

mod bench_suite_cli;
mod catalog_cli;
mod cli_args;
mod consent;
mod doctor_cli;
mod live;
mod model_pack_cli;
mod native_segment_cli;
mod progress;
mod pull_cli;

use catalog_cli::*;
use cli_args::*;
use doctor_cli::*;
use model_pack_cli::*;
use native_segment_cli::*;

const OPENASR_FFMPEG_BIN: &str = "OPENASR_FFMPEG_BIN";
const OPENASR_CATALOG_SIGNING_KEY_SEED_HEX: &str = "OPENASR_CATALOG_SIGNING_KEY_SEED_HEX";
const UNSET_VALUE: &str = "<unset>";

/// Prints the error and exits with its [`consent::ExitCode`] when one is
/// attached (a scriptable failure-class contract), or the generic `1` otherwise.
/// clap keeps its own `2` for usage/argument errors.
fn exit_with_error(error: &anyhow::Error) -> ! {
    eprintln!("Error: {error}");
    let code = error
        .downcast_ref::<consent::CliExit>()
        .map(|exit| exit.code as i32)
        .unwrap_or(1);
    std::process::exit(code);
}

#[cfg(windows)]
fn main() {
    let handle = std::thread::Builder::new()
        .name("openasr-cli-main".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(windows_main_with_expanded_stack)
        .expect("failed to spawn OpenASR CLI main thread");

    if let Err(panic) = handle.join() {
        std::panic::resume_unwind(panic);
    }
}

#[cfg(windows)]
fn windows_main_with_expanded_stack() {
    // Make the console UTF-8 before any output so non-ASCII (Chinese model
    // names, transcripts, paths) renders instead of mojibake.
    set_console_utf8();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("Error: failed to initialize async runtime: {error}");
            std::process::exit(1);
        }
    };

    if let Err(error) = runtime.block_on(run()) {
        exit_with_error(&error);
    }
}

/// Switch the Windows console to UTF-8 so non-ASCII output (Chinese model
/// names, transcripts, paths) renders correctly instead of mojibake. Without
/// it the console uses the legacy OEM/ANSI code page (e.g. 936/GBK on zh-CN
/// Windows), which garbles the UTF-8 bytes emitted by the C engine's stdio and
/// any redirected output. Equivalent to `chcp 65001`; this persists for the
/// console session, as `chcp` does. Best-effort: harmlessly returns 0 when
/// stdout is not a console (redirected to a file/pipe).
#[cfg(windows)]
fn set_console_utf8() {
    const CP_UTF8: u32 = 65001;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetConsoleOutputCP(code_page_id: u32) -> i32;
        fn SetConsoleCP(code_page_id: u32) -> i32;
    }
    // SAFETY: each takes a code-page id and returns BOOL; no pointers and no
    // resource we must release.
    unsafe {
        let _ = SetConsoleOutputCP(CP_UTF8);
        let _ = SetConsoleCP(CP_UTF8);
    }
}

#[cfg(not(windows))]
#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        exit_with_error(&error);
    }
}

async fn run() -> Result<()> {
    match Cli::parse().command {
        Command::List => pull_cli::list_installed(),
        Command::Search { query } => search_models(query.as_deref()),
        Command::Pull {
            reference,
            quant,
            size,
            catalog_url,
            source,
            accept_license,
            from,
        } => tokio::task::spawn_blocking(move || {
            pull_cli::pull(PullCommandOptions {
                reference: &reference,
                quant: quant.as_deref(),
                size: size.as_deref(),
                catalog_url: catalog_url.as_deref(),
                source: source.as_deref(),
                accept_license,
                from: from.as_deref(),
            })
        })
        .await
        .context("openasr pull worker task failed")?,
        Command::Rm { id } => pull_cli::remove_installed(&id),
        Command::Config { command } => config_command(command),
        Command::Doctor => doctor(),
        Command::Verify { path } => model_pack_cli::validate_model_pack_path_command(&path),
        Command::Show { target } => show_model(&target),
        Command::ModelPack { command } => model_pack_command(command),
        Command::GgufCParserProbe { path } => {
            let output = openasr_core::render_gguf_c_parser_sandbox_child_output(&path)?;
            println!("{output}");
            Ok(())
        }
        Command::SignCatalogManifest {
            catalog,
            out,
            epoch,
            catalog_url,
            key_id,
            print_public_key,
        } => sign_catalog_manifest_command(
            &catalog,
            &out,
            epoch,
            catalog_url.as_deref(),
            &key_id,
            print_public_key,
        ),
        Command::Transcribe {
            inputs,
            formats,
            model,
            backend,
            ffmpeg_bin,
            diarize,
            speakers,
            word_timestamps,
            model_pack,
            adapter,
            output,
            continue_on_error,
            benchmark,
            yes,
            offline,
            longform,
            phrase_bias,
            language_task,
        } => transcribe(TranscribeCommandOptions {
            inputs: &inputs,
            formats: &formats,
            model: model.as_deref(),
            backend_kind: backend,
            runtime_paths: RuntimePathOverrides { ffmpeg_bin },
            diarize,
            speakers,
            word_timestamps,
            model_pack: model_pack.as_deref(),
            adapter: adapter.as_deref(),
            output: output.as_deref(),
            continue_on_error,
            benchmark,
            longform,
            phrase_bias,
            language: normalize_language_hint(language_task.language),
            task: language_task.task,
            consent: consent::PullConsent::resolve(yes, offline),
        }),
        Command::Speaker { command } => speaker_command(command),
        Command::BenchSuite {
            config,
            baseline,
            write_baseline,
            format,
            family,
            runs,
            ffmpeg_bin,
            run_single_entry,
        } => bench_suite_cli::bench_suite(BenchSuiteCommandOptions {
            config: &config,
            baseline: baseline.as_deref(),
            write_baseline: write_baseline.as_deref(),
            format,
            family: family.as_deref(),
            runs,
            run_single_entry: run_single_entry.as_deref(),
            runtime_paths: RuntimePathOverrides { ffmpeg_bin },
        }),
        Command::Live {
            source,
            list_devices,
            device,
            input_file,
            model,
            backend,
            model_pack,
            format,
            max_seconds,
            max_utterances,
            frame_duration_ms,
            speech_start_ms,
            speech_stop_ms,
            pre_roll_ms,
            max_utterance_ms,
            no_speech_timeout_ms,
            energy_threshold,
            partial_interval_ms,
            partial_window_ms,
            diarize,
            save,
            save_join_segments,
            save_suggest_title,
            obs_text_file,
            obs_max_lines,
            obs_clear_on_start,
            obs_clear_on_stop,
            markdown_note,
            markdown_append,
            markdown_title,
            markdown_suggest_title,
            ffmpeg_bin,
            yes,
            offline,
        } => {
            live::run_live(live::LiveCommandOptions {
                source,
                list_devices,
                device,
                input_file,
                model: model.as_deref(),
                backend,
                model_pack: model_pack.as_deref(),
                output_format: format,
                max_seconds,
                max_utterances,
                frame_duration_ms,
                speech_start_ms,
                speech_stop_ms,
                pre_roll_ms,
                max_utterance_ms,
                no_speech_timeout_ms,
                energy_threshold,
                partial_interval_ms,
                partial_window_ms,
                diarize,
                save_path: save,
                save_join_segments,
                save_suggest_title,
                obs_text_file,
                obs_max_lines,
                obs_clear_on_start,
                obs_clear_on_stop,
                markdown_note_path: markdown_note,
                markdown_append,
                markdown_title,
                markdown_suggest_title,
                runtime_paths: RuntimePathOverrides { ffmpeg_bin },
                consent: consent::PullConsent::resolve(yes, offline),
            })
            .await
        }
        Command::Serve {
            addr,
            tls_self_signed,
            tls_sans,
            pairing_admin_token_env,
            model,
            backend,
            ffmpeg_bin,
            model_pack,
        } => {
            serve(
                addr,
                model.as_deref(),
                backend,
                RuntimePathOverrides { ffmpeg_bin },
                model_pack.as_deref(),
                ServeSecurityOptions {
                    tls_self_signed,
                    tls_sans,
                    pairing_admin_token_env,
                },
            )
            .await
        }
    }
}

fn search_models(query: Option<&str>) -> Result<()> {
    let cards = load_registry(default_registry_dir()).context("Could not load model registry")?;
    let needle = query.map(|q| q.to_ascii_lowercase());
    println!(
        "{:<30} {:<24} {:<10} {:<7} {:<14} {:<10} {:<10} {:<14} {:<28} DISPLAY NAME",
        "ID", "FAMILY", "TAG", "DEFAULT", "BACKEND", "FORMAT", "QUANT", "SIZE", "QUALITY"
    );
    for card in cards {
        if let Some(needle) = needle.as_deref() {
            let hay = format!("{} {} {}", card.id, card.family_name(), card.display_name)
                .to_ascii_lowercase();
            if !hay.contains(needle) {
                continue;
            }
        }
        let family = card.family_name();
        let tag = card.variant_tag().unwrap_or("-");
        let default = if card.is_default_variant() {
            "yes"
        } else {
            "-"
        };
        let format = card.variant_format().unwrap_or("-");
        let quant = card.variant_quantization().unwrap_or("-");
        println!(
            "{:<30} {:<24} {:<10} {:<7} {:<14} {:<10} {:<10} {:<14} {:<28} {}",
            card.id,
            family,
            tag,
            default,
            card.backend,
            format,
            quant,
            card.size,
            card.quality_profile,
            card.display_name
        );
    }
    Ok(())
}

/// Shows details for a model id (its catalog card) or a local `.oasr` pack file.
/// A path ending in `.oasr` is probed via ggml; anything else is treated as a
/// model id and matched against the catalog.
fn show_model(target: &str) -> Result<()> {
    let path = std::path::Path::new(target);
    if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("oasr") {
        return model_pack_cli::inspect_model_pack_path_command(path);
    }
    search_models(Some(target))
}

fn config_command(command: ConfigCommand) -> Result<()> {
    let home = openasr_home()?;
    let mut config = load_config(&home)?;

    match command {
        ConfigCommand::List => print_config(&config),
        ConfigCommand::Get { key } => {
            let key = ConfigKey::from_str(&key)?;
            println!(
                "{}",
                config.get(key).unwrap_or_else(|| UNSET_VALUE.to_string())
            );
        }
        ConfigCommand::Set { key, value } => {
            let key = ConfigKey::from_str(&key)?;
            set_config_value(&mut config, key, value)?;
            save_config(&home, &config)?;
            println!("Set {}.", key.as_str());
        }
        ConfigCommand::Unset { key } => {
            let key = ConfigKey::from_str(&key)?;
            config.unset(key);
            save_config(&home, &config)?;
            println!("Unset {}.", key.as_str());
        }
    }

    Ok(())
}

fn set_config_value(config: &mut OpenAsrConfig, key: ConfigKey, value: String) -> Result<()> {
    if key == ConfigKey::DefaultModel {
        let home = openasr_home()?;
        let cards =
            load_registry(default_registry_dir()).context("Could not load model registry")?;
        let catalog = load_cli_model_catalog(&home)?;
        config.set_with_catalog(key, value, &cards, catalog.as_ref())?;
    } else {
        config.set(key, value, &[])?;
    }
    Ok(())
}

fn sign_catalog_manifest_command(
    catalog: &Path,
    out: &Path,
    epoch: u64,
    catalog_url: Option<&str>,
    key_id: &str,
    print_public_key: bool,
) -> Result<()> {
    let signing_key_seed_hex =
        env::var(OPENASR_CATALOG_SIGNING_KEY_SEED_HEX).with_context(|| {
            format!(
                "{OPENASR_CATALOG_SIGNING_KEY_SEED_HEX} must be set to a 32-byte hex Ed25519 seed"
            )
        })?;

    if print_public_key {
        let public_key = derive_catalog_public_key_hex(&signing_key_seed_hex)
            .context("Could not derive catalog signature public key")?;
        println!("{public_key}");
        return Ok(());
    }

    let catalog_contents = fs::read_to_string(catalog)
        .with_context(|| format!("Could not read catalog JSON '{}'", catalog.display()))?;
    let source_label = catalog.display().to_string();
    let resolved_catalog_url = match catalog_url {
        Some(value) => value.to_string(),
        None => {
            parse_model_catalog(&catalog_contents, &source_label)
                .context("Could not parse catalog JSON before signing")?
                .catalog_url
        }
    };

    let manifest = render_catalog_signature_manifest(
        &catalog_contents,
        &resolved_catalog_url,
        epoch,
        key_id,
        &signing_key_seed_hex,
    )
    .context("Could not render catalog signature manifest")?;
    verify_catalog_signature_manifest(&catalog_contents, &manifest, &resolved_catalog_url)
        .context(
            "Rendered catalog signature manifest did not verify against the built-in public key",
        )?;

    if let Some(parent) = out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .with_context(|| format!("Could not create output directory '{}'", parent.display()))?;
    }
    atomic_write_text(out, &manifest).with_context(|| {
        format!(
            "Could not write catalog signature manifest '{}'",
            out.display()
        )
    })?;
    println!("Wrote catalog signature manifest: {}", out.display());
    Ok(())
}

fn print_config(config: &OpenAsrConfig) {
    for key in [
        ConfigKey::DefaultModel,
        ConfigKey::DefaultBackend,
        ConfigKey::MediaFfmpegBin,
        ConfigKey::DownloadSource,
    ] {
        println!(
            "{}={}",
            key.as_str(),
            config.get(key).unwrap_or_else(|| UNSET_VALUE.to_string())
        );
    }
}

fn doctor() -> Result<()> {
    let home = openasr_home()?;
    let config_file = config_path(&home);
    let config = load_config(&home)?;
    let cards = load_registry(default_registry_dir()).context("Could not load model registry")?;
    let catalog = load_cli_model_catalog(&home)?;
    let default_model = config.default_model.as_deref().unwrap_or(DEFAULT_MODEL_ID);
    let default_backend = config
        .default_backend
        .as_deref()
        .unwrap_or(DEFAULT_BACKEND_ID);

    println!("OpenASR doctor");
    println!();
    println!("OpenASR home: {}", home.display());
    println!("Config file: {}", config_file.display());
    println!("Model registry: ok ({} models)", cards.len());
    println!(
        "Default model: {} ({})",
        default_model,
        if resolve_runtime_model_ref(&cards, catalog.as_ref(), default_model).is_ok() {
            "ok"
        } else {
            "unknown"
        }
    );
    print_quant_preference_doctor(&home, default_model, catalog.as_ref());
    println!(
        "Default backend: {} ({})",
        default_backend,
        if default_backend.parse::<BackendKind>().is_ok() {
            "ok"
        } else if is_retired_backend_id(default_backend) {
            "legacy"
        } else {
            "unknown"
        }
    );
    println!();
    println!("Backends:");
    println!("- mock: ok");
    print_native_doctor();
    println!();
    println!("Runtimes:");
    print_runtime_doctor();
    println!();
    println!("Audio tools:");
    print_ffmpeg_doctor(&config);
    print_optional_audio_tool("ffprobe");
    Ok(())
}
/// Reports the persisted quant preference and the pack the launch resolver
/// would pick for it — the same ladder the desktop launcher and the server's
/// GET /default use, so doctor output matches what actually runs.
fn print_quant_preference_doctor(
    home: &std::path::Path,
    default_model: &str,
    catalog: Option<&openasr_core::ModelCatalog>,
) {
    let Ok(document) = openasr_core::load_config_document(home) else {
        return;
    };
    let preference = &document.preferences.quant_preference;
    let preference_label = match preference {
        openasr_core::QuantPreference::Auto => "auto".to_string(),
        openasr_core::QuantPreference::Pinned { quant } => format!("pinned ({quant})"),
    };
    let packs = openasr_core::list_installed_packs(home).unwrap_or_default();
    let effective = openasr_core::resolve_launch_pack(
        &packs,
        &openasr_core::LaunchPackRequest {
            model_ref: default_model,
            preference,
            catalog,
            host_profile: openasr_core::host_quant_recommendation_profile(),
        },
    );
    match effective {
        Ok(selection) => println!(
            "Quant preference: {preference_label} (effective: {})",
            selection.runtime_model_id
        ),
        Err(_) => println!("Quant preference: {preference_label} (no installed pack)"),
    }
}

fn speaker_command(command: SpeakerCommand) -> Result<()> {
    match command {
        SpeakerCommand::Enroll {
            input,
            name,
            match_similarity,
        } => enroll_speaker(&input, &name, match_similarity),
        SpeakerCommand::Clear => clear_speaker_profiles(),
    }
}

fn clear_speaker_profiles() -> Result<()> {
    let path = openasr_core::diarize::enrollment::voiceprint_store_path()
        .context("Could not determine the OpenASR home directory for the voiceprint store.")?;
    if path.is_file() {
        std::fs::remove_file(&path)
            .with_context(|| format!("Could not remove {}", path.display()))?;
        println!(
            "Removed speaker voice-match profiles at {}.",
            path.display()
        );
    } else {
        println!(
            "No speaker voice-match profiles to remove at {}.",
            path.display()
        );
    }
    Ok(())
}

fn enroll_speaker(input: &Path, name: &str, match_similarity: Option<f32>) -> Result<()> {
    if let Some(similarity) = match_similarity
        && !(0.0..=1.0).contains(&similarity)
    {
        anyhow::bail!("--match-similarity must be between 0 and 1.");
    }
    let path = openasr_core::diarize::enrollment::voiceprint_store_path()
        .context("Could not determine the OpenASR home directory for the voiceprint store.")?;
    let profile = openasr_core::diarize::enrollment::create_profile_from_wav_file(
        input,
        name,
        match_similarity,
    )
    .map_err(|reason| {
        anyhow::anyhow!(
            "Could not create speaker voice match: {reason}.\nEnrollment needs a 16 kHz mono WAV. Convert any audio first:\n  ffmpeg -i {} -ac 1 -ar 16000 -c:a pcm_s16le enroll.wav\nthen: openasr speaker enroll enroll.wav --name \"{name}\"",
            input.display()
        )
    })?;
    let mut store = openasr_core::diarize::enrollment::VoiceprintStore::load(&path)
        .map_err(|reason| anyhow::anyhow!("Could not read speaker voice-match store: {reason}."))?;
    store.add_profile(profile.clone());
    store
        .save(&path)
        .map_err(|reason| anyhow::anyhow!("Could not save speaker voice-match store: {reason}."))?;
    println!(
        "Created speaker voice-match profile '{}' ({}) from {}.\nSaved to {}. Diarized output can use this display name on the next session; run `openasr speaker clear` to remove local profiles.",
        profile.name,
        profile.id,
        input.display(),
        path.display()
    );
    Ok(())
}

/// Treats `--language auto` (or an empty value) as "no hint" so the model
/// auto-detects, matching the documented omit-for-default behavior.
fn normalize_language_hint(language: Option<String>) -> Option<String> {
    language.filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("auto"))
}

/// `transcribe -` reads a WAV stream from stdin into a temp file used as the sole
/// input (audio prep is extension-based, so stdin is treated as WAV). Returns the
/// temp file to keep alive for the run; `-` must be the only input.
fn maybe_read_stdin_to_temp(inputs: &[PathBuf]) -> Result<Option<tempfile::NamedTempFile>> {
    let dash = Path::new("-");
    if !inputs.iter().any(|input| input == dash) {
        return Ok(None);
    }
    if inputs.len() != 1 {
        return Err(consent::CliExit::new(
            consent::ExitCode::InputError,
            "stdin input '-' must be the only input.".to_string(),
        )
        .into());
    }
    let mut temp = tempfile::Builder::new()
        .prefix("openasr-stdin-")
        .suffix(".wav")
        .tempfile()
        .context("Could not create a temporary file for stdin audio")?;
    std::io::copy(&mut std::io::stdin().lock(), temp.as_file_mut()).map_err(|error| {
        consent::CliExit::new(
            consent::ExitCode::InputError,
            format!("Could not read audio from stdin: {error}"),
        )
    })?;
    Ok(Some(temp))
}

fn transcribe(options: TranscribeCommandOptions<'_>) -> Result<()> {
    let home = openasr_home()?;
    let config = load_config(&home)?;
    // `--benchmark` measures plain transcription timing; run_benchmark does not
    // thread the request-shaping flags, so reject them rather than silently
    // ignoring them (fail-closed). Checked before any pack install or network.
    if options.benchmark
        && (options.diarize
            || options.word_timestamps
            || options.adapter.is_some()
            || options.language.is_some()
            || options.task.is_some()
            || !options.phrase_bias.hotwords.is_empty())
    {
        return Err(consent::CliExit::new(
            consent::ExitCode::InputError,
            "--benchmark measures plain transcription timing; remove --diarize, --word-timestamps, --adapter, --hotword, --language, and --task.".to_string(),
        )
        .into());
    }
    // CLI-only consent-pull: native (the default) without an explicit
    // --model-pack ensures the resolved model is installed, pulling it with a
    // visible confirmation when it is missing. The server never does this.
    let backend = resolve_backend(options.backend_kind, &config)?;
    if backend == BackendKind::Native && options.model_pack.is_none() {
        pull_cli::ensure_asr_model_installed(options.model, &config, &options.consent)?;
    }
    let prepared_run = prepare_backend_run(
        if options.benchmark {
            "benchmark"
        } else {
            "transcription"
        },
        options.model,
        options.backend_kind,
        &options.runtime_paths,
        options.model_pack,
        &config,
    )?;
    ensure_cli_diarization_packs_installed(
        prepared_run.backend_kind,
        prepared_run.model_source.model_pack_path.as_deref(),
        options.diarize,
    )?;
    ensure_diarization_supported(
        prepared_run.backend_kind,
        prepared_run.model_source.model_pack_path.as_deref(),
        options.diarize,
    )?;

    // stdin: `transcribe -` reads a WAV stream from stdin into a temp file used
    // as the single input. Kept alive until the end of the run.
    let stdin_temp = maybe_read_stdin_to_temp(options.inputs)?;
    let inputs: Vec<PathBuf> = match &stdin_temp {
        Some(temp) => vec![temp.path().to_path_buf()],
        None => options.inputs.to_vec(),
    };

    // A directory input or more than one input switches to per-file output.
    let per_file_output = inputs.len() > 1 || inputs.iter().any(|path| path.is_dir());
    let (files, skipped) = expand_transcribe_inputs(&inputs)?;
    if files.is_empty() {
        return Err(consent::CliExit::new(
            consent::ExitCode::InputError,
            "No audio inputs found.".to_string(),
        )
        .into());
    }

    if options.benchmark {
        if per_file_output || files.len() != 1 {
            return Err(consent::CliExit::new(
                consent::ExitCode::InputError,
                "--benchmark takes exactly one input file.".to_string(),
            )
            .into());
        }
        return run_benchmark(
            &prepared_run,
            &files[0],
            options
                .formats
                .first()
                .copied()
                .unwrap_or(ResponseFormat::Text),
            options.output,
            &options.longform,
        );
    }

    let phrase_bias = phrase_bias_options_from_cli(&options.phrase_bias)?;
    ensure_phrase_bias_supported(
        prepared_run.backend_kind,
        prepared_run.model_source.model_pack_path.as_deref(),
        phrase_bias.as_ref(),
    )?;

    if per_file_output {
        if options.word_timestamps || options.adapter.is_some() || phrase_bias.is_some() {
            return Err(consent::CliExit::new(
                consent::ExitCode::InputError,
                "--word-timestamps, --adapter, and --phrase-bias apply to a single input only."
                    .to_string(),
            )
            .into());
        }
        let output_dir = options.output.ok_or_else(|| {
            consent::CliExit::new(
                consent::ExitCode::InputError,
                "Multiple inputs (or a directory) require --output <dir>.".to_string(),
            )
        })?;
        return transcribe_many(&prepared_run, &files, output_dir, skipped, &options);
    }

    // Single input: print to stdout or write one --output file.
    let file = files[0].as_path();
    let prepared = openasr_core::prepare_audio_input(
        file,
        &audio_preparation_options(prepared_run.backend_kind, prepared_run.ffmpeg_bin.clone()),
    )
    .map_err(|error| consent::CliExit::new(consent::ExitCode::InputError, error.to_string()))?;
    print_audio_input_notes(prepared.original());
    print_audio_preparation_notes(&prepared);

    let request = TranscriptionRequest::new(prepared.path(), prepared_run.model_source.model_id)
        .with_model_pack_path(prepared_run.model_source.model_pack_path)
        // OADP Phase 0: the adapter rides the request options into the native
        // executor (the OPENASR_ADAPTER env var stays a server-side surface;
        // mutating the process env here would race the tokio workers). The
        // mock backend rejects adapters instead of silently ignoring them.
        .with_adapter_path(options.adapter.map(Path::to_path_buf))
        .with_language(options.language.clone())
        .with_task(options.task)
        .with_display_file_name(
            file.file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string),
        )
        .with_longform(if prepared_run.backend_kind == BackendKind::Native {
            native_longform_options_override_from_cli(&options.longform)?
        } else {
            None
        })
        .with_phrase_bias(phrase_bias)
        .with_diarization(options.diarize)
        .with_diarize_speakers(options.speakers)
        .with_word_timestamps(options.word_timestamps);
    let transcription =
        transcribe_with_backend(prepared_run.backend_kind, request).map_err(|error| {
            consent::CliExit::new(consent::ExitCode::RuntimeFailed, error.to_string())
        })?;
    write_rendered_formats(&transcription, options.formats, file, options.output, false)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_card(id: &str) -> ModelCard {
        ModelCard {
            id: id.to_string(),
            family: None,
            default_variant: None,
            variant: None,
            display_name: id.to_string(),
            backend: "mock".to_string(),
            task: "transcription".to_string(),
            languages: vec!["en".to_string()],
            size: "tiny".to_string(),
            recommended_hardware: "CPU".to_string(),
            license: "MIT".to_string(),
            features: vec!["transcription".to_string()],
            quality_profile: "fastest".to_string(),
            source: "OpenAI Whisper".to_string(),
        }
    }

    #[test]
    fn parses_supported_backend_values() {
        assert_eq!(parse_backend_kind("mock"), Ok(BackendKind::Mock));
        assert_eq!(parse_backend_kind("native"), Ok(BackendKind::Native));
        assert!(parse_backend_kind("sensevoice-onnx").is_err());
        assert!(parse_backend_kind("whisper.cpp").is_err());
    }

    #[test]
    fn rejects_unknown_backend_value() {
        let error = parse_backend_kind("not-a-backend").unwrap_err();
        assert!(error.contains("Unsupported backend 'not-a-backend'"));
    }

    #[test]
    fn resolves_default_transcribe_model_from_config() {
        let cards = vec![
            test_card("whisper-small"),
            test_card("whisper-large-v3-turbo"),
        ];
        let config = OpenAsrConfig {
            default_model: Some("whisper-small".to_string()),
            ..OpenAsrConfig::default()
        };

        let card = resolve_transcribe_model(&cards, None, &config).unwrap();

        assert_eq!(card.id, "whisper-small");
    }

    #[test]
    fn cli_model_overrides_config_default_model() {
        let cards = vec![
            test_card("whisper-small"),
            test_card("whisper-large-v3-turbo"),
        ];
        let config = OpenAsrConfig {
            default_model: Some("whisper-small".to_string()),
            ..OpenAsrConfig::default()
        };

        let card =
            resolve_transcribe_model(&cards, Some("whisper-large-v3-turbo"), &config).unwrap();

        assert_eq!(card.id, "whisper-large-v3-turbo");
    }

    #[test]
    fn removed_whisper_tiny_default_model_is_unknown() {
        let cards = vec![
            test_card("whisper-small"),
            test_card("whisper-large-v3-turbo"),
        ];
        let config = OpenAsrConfig {
            default_model: Some("whisper-tiny".to_string()),
            ..OpenAsrConfig::default()
        };

        let error = resolve_transcribe_model(&cards, None, &config).unwrap_err();
        assert!(error.to_string().contains("Unknown model: whisper-tiny"));
    }

    #[test]
    fn removed_whisper_family_default_model_is_unknown() {
        let cards = vec![
            test_card("whisper-small"),
            test_card("whisper-large-v3-turbo"),
        ];
        let config = OpenAsrConfig {
            default_model: Some("whisper-tiny.en".to_string()),
            ..OpenAsrConfig::default()
        };

        let error = resolve_transcribe_model(&cards, None, &config).unwrap_err();
        assert!(error.to_string().contains("Unknown model: whisper-tiny.en"));
    }

    #[test]
    fn unknown_tagged_model_refs_fail_fast() {
        let cards = vec![
            test_card("whisper-small"),
            test_card("whisper-large-v3-turbo"),
        ];
        for alias in [
            "unknown-family:q4",
            "unknown-family:q5",
            "unknown-family:onnx",
        ] {
            let config = OpenAsrConfig {
                default_model: Some(alias.to_string()),
                ..OpenAsrConfig::default()
            };
            let error = resolve_transcribe_model(&cards, None, &config).unwrap_err();
            assert!(error.to_string().contains("Unknown model"), "{alias}");
        }
    }

    #[test]
    fn unknown_saved_default_model_still_fails_fast() {
        let cards = vec![
            test_card("whisper-small"),
            test_card("whisper-large-v3-turbo"),
        ];
        let config = OpenAsrConfig {
            default_model: Some("not-a-model".to_string()),
            ..OpenAsrConfig::default()
        };

        let error = resolve_transcribe_model(&cards, None, &config).unwrap_err();
        assert!(error.to_string().contains("Unknown model: not-a-model"));
    }

    #[test]
    fn accepts_native_default_backend_from_saved_config() {
        // `native` is the default backend now: with no explicit `--backend`,
        // transcription resolves an installed pack by model id (and the CLI
        // consent-pulls a missing one), so native no longer needs to be passed
        // explicitly and is a valid saved default.
        let config = OpenAsrConfig {
            default_backend: Some("native".to_string()),
            ..OpenAsrConfig::default()
        };

        assert_eq!(resolve_backend(None, &config).unwrap(), BackendKind::Native);
    }

    #[test]
    fn legacy_default_backend_from_saved_config_fails_closed() {
        let config = OpenAsrConfig {
            default_backend: Some("whisper.cpp".to_string()),
            ..OpenAsrConfig::default()
        };

        let error = resolve_backend(None, &config).unwrap_err().to_string();
        assert!(error.contains("retired and no longer executable"));
    }

    #[test]
    fn unknown_default_backend_from_saved_config_still_fails_fast() {
        let config = OpenAsrConfig {
            default_backend: Some("mokk".to_string()),
            ..OpenAsrConfig::default()
        };

        let error = resolve_backend(None, &config).unwrap_err().to_string();
        assert!(error.contains("Unsupported backend 'mokk'"));
    }

    #[test]
    fn cli_backend_overrides_config_default_backend() {
        let config = OpenAsrConfig {
            default_backend: Some("native".to_string()),
            ..OpenAsrConfig::default()
        };

        assert_eq!(
            resolve_backend(Some(BackendKind::Mock), &config).unwrap(),
            BackendKind::Mock
        );
    }

    #[test]
    fn rejects_unknown_transcribe_model_with_friendly_message() {
        let error = resolve_transcribe_model(&[], Some("not-a-model"), &OpenAsrConfig::default())
            .unwrap_err();
        let message = error.to_string();

        assert!(message.contains("Unknown model: not-a-model"));
        assert!(message.contains("Run `openasr list` to see available models."));
    }

    #[test]
    fn model_pack_requires_native_backend_for_shared_cli_resolution() {
        let error = resolve_model_source_for_backend(
            "benchmark",
            None,
            BackendKind::Mock,
            Some(Path::new("model.gguf")),
            &OpenAsrConfig::default(),
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("--model-pack is only supported with --backend native"));
    }

    #[test]
    fn native_model_source_resolution_without_model_uses_auto_sentinel() {
        let temp = tempfile::tempdir().unwrap();
        let pack_root = temp.path().join("invalid model id!!.gguf");
        fs::write(&pack_root, b"GGUFpayload").unwrap();

        let source = resolve_model_source_for_backend(
            "transcription",
            None,
            BackendKind::Native,
            Some(&pack_root),
            &OpenAsrConfig::default(),
        )
        .expect("native local source should resolve without eager model-id probing");
        assert_eq!(source.model_id, NATIVE_RUNTIME_MODEL_ID_AUTO);
        assert_eq!(source.model_pack_path, Some(pack_root));
    }

    #[test]
    fn mock_model_source_resolution_uses_local_catalog_aliases() {
        let source = resolve_model_source_for_backend(
            "transcription",
            Some("qwen:q8"),
            BackendKind::Mock,
            None,
            &OpenAsrConfig::default(),
        )
        .expect("local catalog should resolve qwen alias for mock backend");

        assert_eq!(source.model_id, "qwen3-asr-0.6b");
        assert_eq!(source.model_pack_path, None);
    }

    #[test]
    fn native_model_source_resolution_uses_local_catalog_aliases_when_available() {
        let temp = tempfile::tempdir().unwrap();
        let pack_root = temp.path().join("qwen3-asr-0.6b-q8_0.gguf");
        fs::write(&pack_root, b"GGUFpayload").unwrap();

        let source = resolve_model_source_for_backend(
            "transcription",
            Some("qwen:q8"),
            BackendKind::Native,
            Some(&pack_root),
            &OpenAsrConfig::default(),
        )
        .expect("local catalog should resolve qwen alias for native backend");

        assert_eq!(source.model_id, "qwen3-asr-0.6b:q8_0");
        assert_eq!(source.model_pack_path, Some(pack_root));
    }

    #[test]
    fn benchmark_flag_accepts_native_model_pack() {
        let cli = Cli::try_parse_from([
            "openasr",
            "transcribe",
            "--benchmark",
            "--backend",
            "native",
            "--model-pack",
            "model.gguf",
            "audio.wav",
        ])
        .unwrap();

        let Command::Transcribe {
            benchmark,
            backend,
            model_pack,
            inputs,
            ..
        } = cli.command
        else {
            panic!("expected transcribe command");
        };

        assert!(benchmark);
        assert_eq!(backend, Some(BackendKind::Native));
        assert_eq!(model_pack, Some(PathBuf::from("model.gguf")));
        assert_eq!(inputs, vec![PathBuf::from("audio.wav")]);
    }

    #[test]
    fn transcribe_cli_accepts_repeated_hotwords_and_boost() {
        let cli = Cli::try_parse_from([
            "openasr",
            "transcribe",
            "--hotword",
            "OpenASR Core",
            "--hotword",
            "Qwen",
            "--hotword-boost",
            "3.5",
            "audio.wav",
        ])
        .unwrap();

        let Command::Transcribe {
            phrase_bias,
            inputs,
            ..
        } = cli.command
        else {
            panic!("expected transcribe command");
        };

        assert_eq!(inputs, vec![PathBuf::from("audio.wav")]);
        assert_eq!(
            phrase_bias.hotwords,
            vec!["OpenASR Core".to_string(), "Qwen".to_string()]
        );
        assert_eq!(phrase_bias.hotword_boost, Some(3.5));
    }

    #[test]
    fn rejects_directory_input_with_friendly_message() {
        let error = openasr_core::validate_audio_input(Path::new(".")).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("Input path is a directory: ."));
        assert!(message.contains("Please provide a valid audio or video file path."));
    }

    #[test]
    fn live_defaults_source_to_mic() {
        let cli = Cli::try_parse_from(["openasr", "live"]).expect("live parses with no --source");
        let Command::Live { source, .. } = cli.command else {
            panic!("expected live command");
        };
        assert_eq!(source, crate::live::LiveSource::Mic);
    }

    #[test]
    fn default_model_ref_matches_documented_constant() {
        // No --model and no saved default resolves to the built-in default,
        // which must stay the documented qwen3-asr-0.6b (guards code/doc drift).
        assert_eq!(
            selected_model_ref(None, &OpenAsrConfig::default(), &[]),
            "qwen3-asr-0.6b"
        );
        assert_eq!(DEFAULT_MODEL_ID, "qwen3-asr-0.6b");
    }

    #[test]
    fn language_auto_and_empty_normalize_to_no_hint() {
        assert_eq!(normalize_language_hint(None), None);
        assert_eq!(normalize_language_hint(Some("auto".to_string())), None);
        assert_eq!(normalize_language_hint(Some("AUTO".to_string())), None);
        assert_eq!(normalize_language_hint(Some(String::new())), None);
        assert_eq!(
            normalize_language_hint(Some("en".to_string())),
            Some("en".to_string())
        );
    }
}
