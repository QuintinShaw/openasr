use super::*;
use openasr_core::NativeAsrModelAdapter;
use openasr_core::TranscriptionTask;
use openasr_core::{batch_output_path, render_transcription};

/// Expands transcribe inputs: a directory is scanned for supported audio/video
/// files; a plain file passes through. Returns the flat file list plus the count
/// of directory entries skipped as unsupported.
pub(super) fn expand_transcribe_inputs(inputs: &[PathBuf]) -> Result<(Vec<PathBuf>, usize)> {
    let mut files = Vec::new();
    let mut skipped = 0;
    for input in inputs {
        if input.is_dir() {
            let discovered = discover_batch_inputs(input)?;
            skipped += discovered.skipped_count;
            files.extend(discovered.files.into_iter().map(|item| item.input_path));
        } else {
            files.push(input.clone());
        }
    }
    Ok((files, skipped))
}

/// Transcribes multiple inputs into `output_dir`, one transcript file per input,
/// then prints a summary. With `continue_on_error`, per-file failures are
/// collected and reported instead of stopping at the first.
pub(super) fn transcribe_many(
    prepared_run: &PreparedBackendRun,
    files: &[PathBuf],
    output_dir: &Path,
    skipped: usize,
    options: &TranscribeCommandOptions<'_>,
) -> Result<()> {
    ensure_batch_output_dir(output_dir)?;
    let longform = if prepared_run.backend_kind == BackendKind::Native {
        native_longform_options_override_from_cli(&options.longform)?
    } else {
        None
    };
    let context = BatchRunContext {
        output_dir,
        formats: options.formats,
        model_id: &prepared_run.model_source.model_id,
        model_pack_path: prepared_run.model_source.model_pack_path.clone(),
        backend_kind: prepared_run.backend_kind,
        ffmpeg_bin: prepared_run.ffmpeg_bin.clone(),
        ffmpeg_bin_explicit: prepared_run.ffmpeg_bin_explicit,
        longform,
        diarize: options.diarize,
        speakers: options.speakers,
        language: options.language.clone(),
        task: options.task,
    };

    let mut outputs = Vec::new();
    let mut failures = Vec::new();
    for file in files {
        match transcribe_batch_item(file, &context) {
            Ok(output) => outputs.push(output),
            Err(error) if options.continue_on_error => failures.push(BatchFailure {
                input_path: file.clone(),
                error: error.to_string(),
            }),
            Err(error) => {
                bail!(
                    "Transcription failed for {}: {}\nCompleted outputs from earlier files were preserved. The failing file output was not written unless a previous final output already existed.",
                    file.display(),
                    error
                );
            }
        }
    }

    // Show the directory the user actually gave when it was a single directory;
    // for multiple explicit files fall back to the first file's parent.
    let input_dir = match options.inputs {
        [single] if single.is_dir() => single.clone(),
        _ => files
            .first()
            .and_then(|file| file.parent())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    };
    let summary = BatchSummary {
        input_dir,
        output_dir: output_dir.to_path_buf(),
        format: options
            .formats
            .first()
            .copied()
            .unwrap_or(ResponseFormat::Text),
        model: prepared_run.model_source.model_id.clone(),
        backend: prepared_run.backend_kind.to_string(),
        files_found: files.len(),
        files_transcribed: outputs.len(),
        files_skipped: skipped,
        files_failed: failures.len(),
        outputs,
        failures,
    };
    print!("{}", render_batch_summary(&summary));
    if summary.files_failed > 0 {
        bail!(
            "Completed with {} failed file(s). See the summary above.",
            summary.files_failed
        );
    }
    Ok(())
}

pub(super) fn transcribe_batch_item(
    input_path: &Path,
    context: &BatchRunContext<'_>,
) -> Result<BatchOutput> {
    let prepared = openasr_core::prepare_audio_input(
        input_path,
        &audio_preparation_options(
            context.backend_kind,
            context.ffmpeg_bin.clone(),
            context.ffmpeg_bin_explicit,
        ),
    )?;
    print_audio_input_notes(prepared.original());
    print_audio_preparation_notes(&prepared);
    let request = TranscriptionRequest::new(prepared.path(), context.model_id)
        .with_model_pack_path(context.model_pack_path.clone())
        .with_language(context.language.clone())
        .with_task(context.task)
        .with_longform(context.longform.clone())
        .with_display_file_name(
            input_path
                .file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string),
        )
        .with_diarization(context.diarize)
        .with_diarize_speakers(context.speakers);
    let transcription = transcribe_with_backend(context.backend_kind, request)?;
    let written = write_rendered_formats(
        &transcription,
        context.formats,
        input_path,
        Some(context.output_dir),
        true,
    )?;
    Ok(BatchOutput {
        input_path: input_path.to_path_buf(),
        output_path: written
            .into_iter()
            .next()
            .unwrap_or_else(|| context.output_dir.to_path_buf()),
    })
}

pub(super) fn ensure_batch_output_dir(output_dir: &Path) -> Result<()> {
    match fs::metadata(output_dir) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => bail!(
            "Batch output path is not a directory: {}\nPlease provide a directory path for batch transcript files.",
            output_dir.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            fs::create_dir_all(output_dir).map_err(|error| {
                anyhow::anyhow!(
                    "Could not create batch output directory: {}\nPlease choose a writable output directory. Details: {error}",
                    output_dir.display()
                )
            })
        }
        Err(error) => Err(anyhow::anyhow!(
            "Could not read batch output directory: {}\nPlease check the path and directory permissions. Details: {error}",
            output_dir.display()
        )),
    }
}

/// Maps the transcription `--format` onto the benchmark report's own format.
fn benchmark_format_from_response_format(format: ResponseFormat) -> BenchmarkFormat {
    match format {
        ResponseFormat::Json | ResponseFormat::VerboseJson => BenchmarkFormat::Json,
        ResponseFormat::Markdown => BenchmarkFormat::Markdown,
        _ => BenchmarkFormat::Text,
    }
}

/// Runs one transcription and prints timing metadata (elapsed, audio duration,
/// real-time factor) instead of the transcript. Backs `transcribe --benchmark`.
pub(super) fn run_benchmark(
    prepared_run: &PreparedBackendRun,
    file: &Path,
    format: ResponseFormat,
    output: Option<&Path>,
    longform_cli: &NativeLongFormCliOptions,
) -> Result<()> {
    let prepared = openasr_core::prepare_audio_input(
        file,
        &audio_preparation_options(
            prepared_run.backend_kind,
            prepared_run.ffmpeg_bin.clone(),
            prepared_run.ffmpeg_bin_explicit,
        ),
    )?;
    print_audio_input_notes(prepared.original());
    print_audio_preparation_notes(&prepared);

    let longform = if prepared_run.backend_kind == BackendKind::Native {
        native_longform_options_override_from_cli(longform_cli)?
    } else {
        None
    };
    let request =
        TranscriptionRequest::new(prepared.path(), prepared_run.model_source.model_id.clone())
            .with_model_pack_path(prepared_run.model_source.model_pack_path.clone())
            .with_longform(longform)
            .with_display_file_name(
                file.file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_string),
            )
            // `--benchmark` measures plain ASR decode timing; punctuation
            // restoration is an optional post-process (like diarization and
            // word-timestamp alignment, neither of which this request enables
            // either) and would silently skew the real-time factor for an
            // unpunctuated model with the FireRedPunc pack installed.
            .with_punctuation(false);
    let started = Instant::now();
    let transcription = transcribe_with_backend(prepared_run.backend_kind, request)?;
    let elapsed = started.elapsed();

    let audio_duration_seconds = prepared.original().duration_seconds.or_else(|| {
        prepared
            .is_converted()
            .then(|| openasr_core::probe_wav_duration(prepared.path()))
            .flatten()
    });
    let real_time_factor = audio_duration_seconds
        .filter(|duration| *duration > 0.0)
        .map(|duration| elapsed.as_secs_f64() / duration);
    let longform_metrics = transcription.longform.as_ref();
    let bench_format = benchmark_format_from_response_format(format);
    let result = BenchmarkResult {
        input: file.display().to_string(),
        model: prepared_run.model_source.model_id.clone(),
        backend: prepared_run.backend_kind.to_string(),
        elapsed_ms: elapsed.as_millis(),
        audio_duration_seconds,
        real_time_factor,
        text_length: transcription.text.chars().count(),
        segment_count: transcription.segments.len(),
        chunk_count: longform_metrics.map(|value| value.chunk_count),
        skipped_silent_chunks: longform_metrics.map(|value| value.skipped_silent_chunks),
        duplicate_merge_count: longform_metrics.map(|value| value.duplicate_merge_count),
        provenance: longform_metrics.map(|value| value.provenance.clone()),
        output_format: bench_format.to_string(),
    };

    let rendered =
        render_benchmark(&result, bench_format).context("Could not render benchmark output")?;
    write_rendered_output(&rendered, output)?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResolvedModelSource {
    pub(super) model_id: String,
    pub(super) model_pack_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PreparedBackendRun {
    pub(super) backend_kind: BackendKind,
    pub(super) model_source: ResolvedModelSource,
    pub(super) ffmpeg_bin: Option<PathBuf>,
    /// Whether `ffmpeg_bin` came from an explicit user choice (CLI flag, env
    /// var, or config) rather than PATH auto-discovery -- see
    /// `AudioPreparationOptions::with_ffmpeg_bin_explicit`.
    pub(super) ffmpeg_bin_explicit: bool,
}

pub(super) fn resolve_model_source_for_backend(
    command_label: &str,
    model: Option<&str>,
    backend_kind: BackendKind,
    model_pack: Option<&Path>,
    config: &OpenAsrConfig,
) -> Result<ResolvedModelSource> {
    let catalog = load_cli_model_catalog(&openasr_home()?)?;

    if backend_kind != BackendKind::Native {
        if model_pack.is_some() {
            bail!(
                "--model-pack is only supported with --backend native.\nUse --backend native, or remove --model-pack."
            );
        }
        let cards = runtime_registry(catalog.as_ref()).context("Could not load model registry")?;
        let model_ref = selected_model_ref(model, config, &cards);
        let model_id = find_runtime_model_id(&cards, catalog.as_ref(), &model_ref)?;
        return Ok(ResolvedModelSource {
            model_id,
            model_pack_path: None,
        });
    }

    // Native: an explicit --model-pack is the advanced escape hatch; otherwise
    // resolve an installed pack by model id. This path NEVER pulls -- the CLI
    // transcribe/live handlers run the consent-pull before reaching here, while
    // the server stays fail-closed (a missing model is an error, not a download).
    let model_pack_root = match model_pack {
        Some(path) => validate_local_native_model_pack_path(path)
            .map_err(|error| anyhow!("Native model-pack path rejected: {error}"))?,
        None => resolve_installed_native_pack(model, config, catalog.as_ref())?,
    };
    let model_id = if let Some(model_ref) = model {
        let normalized_model_ref = model_ref.trim();
        parse_model_ref(normalized_model_ref).map_err(|error| {
            anyhow::anyhow!(
                "Model '{model_ref}' is not a valid model id for native GGUF local-source {command_label}: {error}"
            )
        })?;
        // Resolve catalog aliases (e.g. `qwen:q8`) to the canonical runtime id so
        // the alias-blind native matcher accepts the request. The reported
        // identity still derives from pack metadata downstream.
        let cards = runtime_registry(catalog.as_ref()).context("Could not load model registry")?;
        match resolve_runtime_model_ref(&cards, catalog.as_ref(), normalized_model_ref) {
            Ok(resolved) => resolved.runtime_model_id,
            Err(error) if runtime_resolution_unknown_model(&error) => {
                normalized_model_ref.to_owned()
            }
            Err(error) => return Err(anyhow::anyhow!(error)),
        }
    } else {
        NATIVE_RUNTIME_MODEL_ID_AUTO.to_string()
    };
    Ok(ResolvedModelSource {
        model_id,
        model_pack_path: Some(model_pack_root),
    })
}

/// Resolves the installed `.oasr` pack for a model id (the resolved default when
/// `model` is `None`), or `Ok(None)` when no matching pack is installed yet (a
/// normal state right after a fresh install, before the user has pulled any
/// model). Genuine environment/registry errors (unreadable `OPENASR_HOME`,
/// corrupt registry, ...) still return `Err`. Never pulls either way.
///
/// An explicit `model` reference is a CLI-local concern (not "the default") and
/// is matched directly against installed packs with `QuantPreference::Auto`.
/// With no explicit reference, resolving `config.default_model` against
/// installed packs -- including the `default.json` pointer fallback and
/// `Pinned` quant recovery -- is delegated to `openasr_core::default_selection`,
/// the single authority also used by the server; only the
/// no-persisted-default-at-all fallback to `DEFAULT_MODEL_ID` stays here, since
/// that bare-invocation convention is CLI-specific, not part of "the default".
fn resolve_installed_native_pack_opt(
    model: Option<&str>,
    // `default_selection::resolve_with_catalog` reads `config.default_model`
    // straight off disk (the single-authority contract requires re-reading, not
    // trusting a possibly-stale in-memory copy); kept for signature parity with
    // `resolve_installed_native_pack`, whose error message still needs it.
    _config: &OpenAsrConfig,
    catalog: Option<&openasr_core::ModelCatalog>,
) -> Result<Option<PathBuf>> {
    let home = openasr_home()?;
    if let Some(model_ref) = model {
        return resolve_launch_pack_path(&home, model_ref, catalog);
    }

    use openasr_core::default_selection::DefaultModelResolution;
    match openasr_core::default_selection::resolve_with_catalog(&home, catalog)? {
        DefaultModelResolution::Installed(pack) => Ok(Some(pack.path)),
        DefaultModelResolution::NotInstalled(_) => Ok(None),
        DefaultModelResolution::Unset => resolve_launch_pack_path(&home, DEFAULT_MODEL_ID, catalog),
    }
}

fn resolve_launch_pack_path(
    home: &Path,
    model_ref: &str,
    catalog: Option<&openasr_core::ModelCatalog>,
) -> Result<Option<PathBuf>> {
    let packs = openasr_core::list_installed_packs(home)?;
    let request = openasr_core::LaunchPackRequest {
        model_ref,
        preference: &openasr_core::QuantPreference::Auto,
        catalog,
        host_profile: openasr_core::host_quant_recommendation_profile(),
    };
    match openasr_core::resolve_launch_pack(&packs, &request) {
        Ok(selection) => Ok(Some(selection.pack.path)),
        Err(_) => Ok(None),
    }
}

/// Resolves the installed `.oasr` pack for a model id (the resolved default when
/// `model` is `None`). Never pulls: a missing model is a fail-closed error here.
/// The CLI transcribe/live handlers ensure the pack is installed (consent-pull)
/// before this runs; the server relies on this staying download-free.
pub(super) fn resolve_installed_native_pack(
    model: Option<&str>,
    config: &OpenAsrConfig,
    catalog: Option<&openasr_core::ModelCatalog>,
) -> Result<PathBuf> {
    let model_ref = selected_model_ref(model, config, &[]);
    resolve_installed_native_pack_opt(model, config, catalog)?.ok_or_else(|| {
        anyhow!(
            "Model '{model_ref}' is not installed.\nRun: openasr pull {model_ref}\n(Or pass --model-pack <local.oasr> to run a specific local pack file.)"
        )
    })
}

pub(super) fn prepare_backend_run(
    command_label: &str,
    model: Option<&str>,
    backend_kind: Option<BackendKind>,
    runtime_paths: &RuntimePathOverrides,
    model_pack: Option<&Path>,
    config: &OpenAsrConfig,
) -> Result<PreparedBackendRun> {
    let backend_kind = resolve_backend(backend_kind, config)?;
    let model_source =
        resolve_model_source_for_backend(command_label, model, backend_kind, model_pack, config)?;
    let ffmpeg_bin_explicit =
        resolve_explicit_ffmpeg_bin(runtime_paths.ffmpeg_bin.clone(), config).is_some();
    let ffmpeg_bin = resolve_ffmpeg_bin(runtime_paths.ffmpeg_bin.clone(), config);

    Ok(PreparedBackendRun {
        backend_kind,
        model_source,
        ffmpeg_bin,
        ffmpeg_bin_explicit,
    })
}

/// Resolves the model source for `serve`. Unlike `resolve_model_source_for_backend`
/// (used by `transcribe`/`live`, which run consent-pull first and so treat a
/// missing pack as fatal), `serve` must come up with zero models installed --
/// that is a normal post-install state, not a startup error. An explicit
/// `--model-pack` escape hatch still fails closed if the path does not
/// validate; only the "nothing installed yet" case degrades to no model bound,
/// which `openasr-server` reports via `/health` and fails closed on a
/// transcription request instead.
pub(super) fn resolve_serve_model_source(
    model: Option<&str>,
    backend_kind: BackendKind,
    model_pack: Option<&Path>,
    config: &OpenAsrConfig,
) -> Result<ResolvedModelSource> {
    if backend_kind != BackendKind::Native {
        return resolve_model_source_for_backend("serve", model, backend_kind, model_pack, config);
    }
    let catalog = load_cli_model_catalog(&openasr_home()?)?;
    let model_pack_root = match model_pack {
        Some(path) => Some(
            validate_local_native_model_pack_path(path)
                .map_err(|error| anyhow!("Native model-pack path rejected: {error}"))?,
        ),
        None => resolve_installed_native_pack_opt(model, config, catalog.as_ref())?,
    };
    let model_id = match &model_pack_root {
        Some(_) => {
            if let Some(model_ref) = model {
                let normalized_model_ref = model_ref.trim();
                parse_model_ref(normalized_model_ref).map_err(|error| {
                    anyhow!(
                        "Model '{model_ref}' is not a valid model id for native GGUF local-source serve: {error}"
                    )
                })?;
                let cards =
                    runtime_registry(catalog.as_ref()).context("Could not load model registry")?;
                match resolve_runtime_model_ref(&cards, catalog.as_ref(), normalized_model_ref) {
                    Ok(resolved) => resolved.runtime_model_id,
                    Err(error) if runtime_resolution_unknown_model(&error) => {
                        normalized_model_ref.to_owned()
                    }
                    Err(error) => return Err(anyhow::anyhow!(error)),
                }
            } else {
                NATIVE_RUNTIME_MODEL_ID_AUTO.to_string()
            }
        }
        // No pack resolved: keep the requested model id (or the auto sentinel)
        // around so the health/status payload can report which model, if any,
        // was asked for.
        None => model
            .map(str::to_owned)
            .unwrap_or_else(|| NATIVE_RUNTIME_MODEL_ID_AUTO.to_string()),
    };
    Ok(ResolvedModelSource {
        model_id,
        model_pack_path: model_pack_root,
    })
}

pub(super) async fn serve(
    addr: SocketAddr,
    model: Option<&str>,
    backend_kind: Option<BackendKind>,
    runtime_paths: RuntimePathOverrides,
    model_pack: Option<&Path>,
    security: ServeSecurityOptions,
) -> Result<()> {
    let home = openasr_home()?;
    // Read the config document once: `config` and `idle_unload` (used below
    // for `idle_unload_after`) both live on it, so reading it a second time
    // further down would be a redundant fs::read + serde_json parse on every
    // serve() startup.
    let config_document = openasr_core::load_config_document(&home)?;
    let config = &config_document.config;
    let backend = resolve_backend(backend_kind, config)?;
    let model_source = resolve_serve_model_source(model, backend, model_pack, config)?;
    if backend == BackendKind::Native
        && let Some(model_pack_path) = model_source.model_pack_path.as_deref()
    {
        let local_model_id = resolve_native_runtime_model_id_from_source(model_pack_path)?;
        // Tolerant matching, not string equality: `model_source.model_id` is the
        // catalog-resolved ref (e.g. `whisper-tiny:q8_0`) while the pack's
        // runtime id is bare (`whisper-tiny`), so equality would reject every
        // catalog-installed pack the daemon is about to serve.
        if let Some(model_ref) = model
            && !openasr_core::native_runtime_model_refs_match(
                &model_source.model_id,
                &local_model_id,
            )
        {
            bail!(
                "Native GGUF local-source serve mode requires --model to match local source id '{}', got '{}' (resolved '{}').\nUse --model {} or omit --model.",
                local_model_id,
                model_ref,
                model_source.model_id,
                local_model_id
            );
        }
    } else if backend == BackendKind::Native {
        eprintln!(
            "openasr-server: no installed native model pack found; starting with no model bound. Install one (openasr pull <model-id>) or install via the desktop model market; transcription requests will fail closed until then."
        );
    }
    let ffmpeg_bin_explicit =
        resolve_explicit_ffmpeg_bin(runtime_paths.ffmpeg_bin.clone(), config).is_some();
    let ffmpeg_bin = resolve_ffmpeg_bin(runtime_paths.ffmpeg_bin.clone(), config);
    let api_key_hashes = if supervised_daemon_launch() {
        // The desktop supervisor's managed daemon (marked by the instance-token
        // env it always sets) has its own trust model: the UI talks to its
        // daemon over loopback without bearer headers, and remote access goes
        // through TLS + pairing. Enforcing user-created API keys here would
        // lock the desktop app out of its own daemon, so keys apply only to
        // manually-launched `openasr serve`.
        Vec::new()
    } else {
        load_active_api_key_hashes()?
    };

    let mut launch_options = serve_launch_options(addr, security, api_key_hashes)?;
    // Persist pairing credentials/revocations under OPENASR_HOME so a paired
    // remote server keeps its devices across the restarts the desktop performs on
    // every daemon start (no-op for the local non-pairing UI daemon).
    launch_options.auth = launch_options
        .auth
        .with_pairing_store(home.join("pairing-registry.json"));
    // Persist the self-signed TLS private key + certificate under OPENASR_HOME,
    // alongside pairing-registry.json, so a --tls-self-signed daemon keeps the
    // same certificate fingerprint (and therefore the same pairing safety code
    // and every already-paired client's TOFU pin) across the restarts the
    // desktop performs on every model switch. No-op when TLS is disabled.
    launch_options.tls_identity_store = Some(home.join("tls-identity.json"));
    // `idle_unload` lives on `Preferences`, on the same document already
    // loaded above as `config_document` -- no second read needed.
    launch_options.idle_unload_after = config_document.preferences.idle_unload.idle_threshold();
    openasr_server::serve_with_launch_options(
        addr,
        openasr_server::ServerRuntime {
            backend,
            ffmpeg_bin,
            ffmpeg_bin_explicit,
            model_pack_path: model_source.model_pack_path,
        },
        launch_options,
    )
    .await
}

/// True when this `serve` process was launched by the desktop supervisor,
/// which always sets the server instance-token env for its managed daemon
/// (`OPENASR_SERVER_INSTANCE_TOKEN`, consumed by `openasr-server` for
/// same-port restart identity).
fn supervised_daemon_launch() -> bool {
    env::var_os("OPENASR_SERVER_INSTANCE_TOKEN")
        .is_some_and(|value| !value.to_string_lossy().trim().is_empty())
}

/// Reads currently-active API key hashes from the local `apikeys.json` store
/// (see `openasr apikey create/list/revoke`). An unreadable store fails
/// closed (serve refuses to start) rather than silently opening loopback
/// access; a missing store is just "no keys yet" and returns empty.
fn load_active_api_key_hashes() -> Result<Vec<String>> {
    let Some(path) = openasr_core::apikeys::api_key_store_path() else {
        return Ok(Vec::new());
    };
    let store = openasr_core::apikeys::ApiKeyStore::load(&path)
        .with_context(|| format!("Could not load API key store at {}", path.display()))?;
    Ok(store.active_token_hashes())
}

#[derive(Debug, Clone, Default)]
pub(super) struct ServeSecurityOptions {
    pub tls_self_signed: bool,
    pub tls_sans: Vec<String>,
    pub pairing_admin_token_env: Option<String>,
}

fn serve_launch_options(
    addr: SocketAddr,
    security: ServeSecurityOptions,
    api_key_hashes: Vec<String>,
) -> Result<openasr_server::ServerLaunchOptions> {
    let tls = if security.tls_self_signed {
        openasr_server::ServerTlsConfig::self_signed(default_tls_subject_alt_names(
            addr,
            &security.tls_sans,
        ))
    } else {
        openasr_server::ServerTlsConfig::Disabled
    };
    let auth = match security
        .pairing_admin_token_env
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        Some(env_name) => {
            let token = env::var(env_name).with_context(|| {
                format!("Could not read pairing administrator token from ${env_name}")
            })?;
            let token = token.trim();
            if token.is_empty() {
                bail!("Pairing administrator token in ${env_name} must not be empty.");
            }
            openasr_server::ServerAuth::pairing(token)
        }
        // Local API keys (`openasr apikey create`) are a loopback-only escape
        // hatch: they let a trusted-but-explicit caller (a coding agent, a
        // script) require a bearer credential even from 127.0.0.1, where the
        // server otherwise trusts every caller by default. They must never
        // relax the non-loopback path, which stays fail-closed on TLS +
        // device pairing regardless of any configured key.
        None if addr.ip().is_loopback() => {
            openasr_server::ServerAuth::from_token_hashes(api_key_hashes)
        }
        None => openasr_server::ServerAuth::disabled(),
    };
    Ok(openasr_server::ServerLaunchOptions {
        auth,
        tls,
        ..Default::default()
    })
}

fn default_tls_subject_alt_names(addr: SocketAddr, configured: &[String]) -> Vec<String> {
    let mut names = configured
        .iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let ip = addr.ip().to_string();
    if !addr.ip().is_unspecified() && !names.iter().any(|name| name == &ip) {
        names.push(ip);
    }
    if addr.ip().is_loopback() && !names.iter().any(|name| name == "localhost") {
        names.push("localhost".to_string());
    }
    names
}

pub(super) fn resolve_native_runtime_model_id_from_source(
    model_pack_root: &Path,
) -> Result<String> {
    let identity = openasr_core::resolve_local_native_runtime_model_identity(model_pack_root, None)
        .map_err(|error| anyhow!("{error}"))?;
    Ok(identity.model_id)
}

pub(crate) fn transcribe_with_backend(
    backend_kind: BackendKind,
    request: TranscriptionRequest,
) -> Result<openasr_core::Transcription> {
    match backend_kind {
        BackendKind::Mock => transcribe_with_mock_backend(request).map_err(Into::into),
        BackendKind::Native => {
            configure_native_cpu_inference_threads();
            NativeBackend.transcribe(request).map_err(Into::into)
        }
    }
}

pub(crate) fn configure_native_cpu_inference_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let Ok(available) = std::thread::available_parallelism() else {
        return;
    };
    let threads = available.get().min(5);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
}

pub(super) fn resolve_ffmpeg_bin(
    cli_path: Option<PathBuf>,
    config: &OpenAsrConfig,
) -> Option<PathBuf> {
    resolve_explicit_ffmpeg_bin(cli_path, config).or_else(|| find_in_path("ffmpeg"))
}

/// Resolves ffmpeg only from explicit user choices (`--ffmpeg-bin`,
/// `OPENASR_FFMPEG_BIN`, or the persisted `media.ffmpeg_bin` config) --
/// excludes PATH auto-discovery. A system that merely happens to have ffmpeg
/// on PATH should not disable the in-process symphonia decode path, so this
/// is what decides `AudioPreparationOptions::with_ffmpeg_bin_explicit`.
pub(super) fn resolve_explicit_ffmpeg_bin(
    cli_path: Option<PathBuf>,
    config: &OpenAsrConfig,
) -> Option<PathBuf> {
    cli_path
        .or_else(|| env_path(OPENASR_FFMPEG_BIN))
        .or_else(|| config.media.ffmpeg_bin.as_ref().map(PathBuf::from))
}

pub(super) fn audio_preparation_options(
    backend: BackendKind,
    ffmpeg_bin: Option<PathBuf>,
    ffmpeg_bin_explicit: bool,
) -> AudioPreparationOptions {
    AudioPreparationOptions::new(backend)
        .with_ffmpeg_bin(ffmpeg_bin)
        .with_ffmpeg_bin_explicit(ffmpeg_bin_explicit)
        .with_native_non_wav_conversion(backend == BackendKind::Native)
}

pub(super) fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub(super) fn find_model<'a>(
    cards: &'a [ModelCard],
    model: &str,
) -> Result<openasr_core::ResolvedModel<'a>> {
    resolve_registry_model_ref(cards, model).map_err(|error| anyhow::anyhow!(error))
}

fn find_runtime_model_id(
    cards: &[ModelCard],
    catalog: Option<&openasr_core::ModelCatalog>,
    model: &str,
) -> Result<String> {
    if let Some(catalog) = catalog {
        match resolve_runtime_model_ref(cards, Some(catalog), model) {
            Ok(resolved) => return Ok(resolved.model_id),
            Err(error) if runtime_resolution_unknown_model(&error) => {}
            Err(error) => return Err(anyhow::anyhow!(error)),
        }
    }
    Ok(find_model(cards, model)?.card.id.clone())
}

fn runtime_resolution_unknown_model(error: &openasr_core::RuntimeModelResolutionError) -> bool {
    matches!(
        error,
        openasr_core::RuntimeModelResolutionError::Catalog(
            openasr_core::CatalogError::UnknownModel { .. }
        ) | openasr_core::RuntimeModelResolutionError::Registry(
            openasr_core::ModelResolutionError::UnknownModel(_)
        )
    )
}
#[cfg(test)]
pub(super) fn resolve_transcribe_model<'a>(
    cards: &'a [ModelCard],
    model: Option<&str>,
    config: &OpenAsrConfig,
) -> Result<&'a ModelCard> {
    Ok(find_model(cards, &selected_model_ref(model, config, cards))?.card)
}

pub(super) fn selected_model_ref(
    model: Option<&str>,
    config: &OpenAsrConfig,
    _cards: &[ModelCard],
) -> String {
    if let Some(model) = model {
        return model.to_string();
    }

    if let Some(config_default) = config.default_model.as_deref() {
        return config_default.to_string();
    }

    DEFAULT_MODEL_ID.to_string()
}

pub(super) fn resolve_backend(
    backend: Option<BackendKind>,
    config: &OpenAsrConfig,
) -> Result<BackendKind> {
    if let Some(backend) = backend {
        return Ok(backend);
    }

    let configured = config
        .default_backend
        .as_deref()
        .unwrap_or(DEFAULT_BACKEND_ID);
    match configured {
        "mock" => Ok(BackendKind::Mock),
        // `native` is now the default: real transcription resolves an installed
        // pack by model id (and the CLI consent-pulls a missing one), so it no
        // longer needs an explicit `--backend native`.
        "native" => Ok(BackendKind::Native),
        other => {
            if is_retired_backend_id(other) {
                Err(anyhow::anyhow!(
                    "Saved default backend '{other}' is retired and no longer executable.\nRun `openasr config set default_backend mock` to migrate your persisted config, or pass `--backend mock` explicitly.",
                ))
            } else {
                parse_backend_kind(other).map_err(anyhow::Error::msg)
            }
        }
    }
}

pub(super) fn is_retired_backend_id(value: &str) -> bool {
    matches!(
        value,
        "whisper.cpp" | "sensevoice-onnx" | "sensevoice.cpp" | "transcribe-rs" | "sherpa-onnx"
    )
}

pub(super) fn ensure_diarization_supported(
    backend: BackendKind,
    model_pack_path: Option<&Path>,
    diarize: bool,
) -> Result<()> {
    if !diarize {
        return Ok(());
    }

    if diarization_supported(backend, model_pack_path) {
        return Ok(());
    }

    Err(anyhow::anyhow!(
        openasr_core::BackendError::DiarizationNotSupported {
            backend: backend_name(backend)
        }
    ))
}

fn diarization_supported(backend: BackendKind, model_pack_path: Option<&Path>) -> bool {
    match (backend, model_pack_path) {
        (BackendKind::Native, Some(pack_path)) => {
            openasr_core::native_runtime_transcription_capabilities_for_path(pack_path)
                .diarization
                .supported
        }
        // The VAD + active speaker-embedder diarization path is model-agnostic,
        // so without a resolved pack path the native answer is exactly "is it installed".
        (BackendKind::Native, None) => openasr_core::diarize::vad_diarization_available(),
        _ => {
            openasr_core::api::backend::TranscriptionBackendCapabilities::for_backend_kind(backend)
                .diarization
                .supported
        }
    }
}

pub(super) fn ensure_cli_diarization_packs_installed(
    backend: BackendKind,
    model_pack_path: Option<&Path>,
    diarize: bool,
) -> Result<()> {
    if !diarize || backend != BackendKind::Native || diarization_supported(backend, model_pack_path)
    {
        return Ok(());
    }

    let home = openasr_home()?;
    let config = load_config(&home)?;
    let catalog = match load_cli_model_catalog(&home)? {
        Some(catalog) => catalog,
        None => openasr_core::load_model_catalog(None, &home)?,
    };
    let installed_packs = openasr_core::list_installed_packs(&home)?;
    let source_chain = openasr_core::resolve_chain(&config.download_source);
    let required_pack = catalog
        .speaker_diarization_required_embedder_pack()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Public catalog does not contain the WeSpeaker speaker-diarization embedder pack."
            )
        })?;

    install_cli_capability_pack_if_missing(
        &installed_packs,
        &catalog,
        required_pack,
        &home,
        &source_chain,
    )?;

    Ok(())
}

/// Whether `--word-timestamps=aligned` requires a backend it does not run on.
/// The alignment refinement pass re-decodes the full file and the finished
/// transcript through a second local pack, which only the native backend
/// supports; approximate (or omitted) timestamps are unaffected.
pub(super) fn ensure_word_timestamps_alignment_supported(
    backend: BackendKind,
    word_timestamps_mode: Option<WordTimestampsMode>,
) -> Result<()> {
    if !matches!(word_timestamps_mode, Some(WordTimestampsMode::Aligned)) {
        return Ok(());
    }
    if backend != BackendKind::Native {
        bail!("--word-timestamps=aligned requires the native backend.");
    }
    Ok(())
}

/// Passing `--word-timestamps=aligned` is itself the consent to install the
/// Qwen3-ForcedAligner-0.6B capability pack, mirroring `--diarize`'s WeSpeaker
/// auto-install above -- `approximate` (or an omitted flag) never touches the
/// network.
pub(super) fn ensure_cli_word_timestamps_pack_installed(
    backend: BackendKind,
    word_timestamps_mode: Option<WordTimestampsMode>,
) -> Result<()> {
    if !matches!(word_timestamps_mode, Some(WordTimestampsMode::Aligned))
        || backend != BackendKind::Native
    {
        return Ok(());
    }

    let home = openasr_home()?;
    let config = load_config(&home)?;
    let catalog = match load_cli_model_catalog(&home)? {
        Some(catalog) => catalog,
        None => openasr_core::load_model_catalog(None, &home)?,
    };
    let installed_packs = openasr_core::list_installed_packs(&home)?;
    let source_chain = openasr_core::resolve_chain(&config.download_source);
    let required_pack = catalog.word_timestamps_forced_aligner_pack().ok_or_else(|| {
        anyhow::anyhow!(
            "Public catalog does not contain a word-timestamps forced-alignment capability pack."
        )
    })?;

    install_cli_capability_pack_if_missing(
        &installed_packs,
        &catalog,
        required_pack,
        &home,
        &source_chain,
    )
}

fn install_cli_capability_pack_if_missing(
    installed_packs: &[openasr_core::InstalledPack],
    catalog: &openasr_core::ModelCatalog,
    model: &openasr_core::CatalogModel,
    home: &Path,
    source_chain: &[openasr_core::DownloadSource],
) -> Result<()> {
    if openasr_core::resolve_installed_pack_reference_with_catalog(
        installed_packs,
        catalog,
        &model.pull_recommended,
    )?
    .is_some()
    {
        return Ok(());
    }
    let resolved = openasr_core::resolve_catalog_pull(
        catalog,
        &openasr_core::CatalogPullRequest {
            reference: model.pull_recommended.clone(),
            quant: None,
            size: None,
        },
    )?;
    install_cli_capability_pack(&resolved, home, source_chain)
}

fn install_cli_capability_pack(
    resolved: &openasr_core::ResolvedCatalogPull,
    home: &Path,
    source_chain: &[openasr_core::DownloadSource],
) -> Result<()> {
    // Reuse the same progress UX as the main `pull` command (indicatif bar on a
    // TTY, plain periodic lines otherwise) instead of a second, weaker
    // hand-rolled renderer that never showed a progress bar for
    // diarization/word-timestamps/punc capability-pack downloads.
    let mut reporter = crate::progress::PullReporter::new(&resolved.pull);
    let progress = |event| reporter.on(event);
    openasr_core::PullModelPackRequest::new(resolved, home)
        .sources(source_chain)
        .execute(progress)?;
    Ok(())
}

pub(super) fn phrase_bias_options_from_cli(
    cli: &PhraseBiasCliOptions,
) -> Result<Option<openasr_core::PhraseBiasConfig>> {
    if cli.hotwords.is_empty() {
        if cli.hotword_boost.is_some() {
            bail!("--hotword-boost requires at least one --hotword.");
        }
        return Ok(None);
    }

    openasr_core::PhraseBiasConfig::from_phrases_with_default_boost(
        cli.hotwords.iter().cloned(),
        cli.hotword_boost,
    )
    .map(Some)
    .map_err(|error| anyhow::anyhow!("Invalid phrase bias CLI options: {error}"))
}

pub(super) fn ensure_phrase_bias_supported(
    backend: BackendKind,
    model_pack_path: Option<&Path>,
    phrase_bias: Option<&openasr_core::PhraseBiasConfig>,
) -> Result<()> {
    if phrase_bias.is_none_or(openasr_core::PhraseBiasConfig::is_empty) {
        return Ok(());
    }

    let capabilities = match (backend, model_pack_path) {
        (BackendKind::Native, Some(pack_path)) => {
            openasr_core::native_runtime_transcription_capabilities_for_path(pack_path)
        }
        _ => {
            openasr_core::api::backend::TranscriptionBackendCapabilities::for_backend_kind(backend)
        }
    };
    if capabilities.phrase_bias.supported {
        return Ok(());
    }

    if backend == BackendKind::Native
        && let Some(pack_path) = model_pack_path
        && let Some(adapter) = openasr_core::native_runtime_model_adapter_for_path(pack_path)
    {
        bail!(
            "--hotword is not supported by native model family '{}' ({}). Omit --hotword/--hotword-boost; the request was rejected instead of silently ignoring phrase_bias.",
            adapter.model_family(),
            adapter.adapter_id()
        );
    }

    Err(anyhow::anyhow!(
        openasr_core::BackendError::PhraseBiasNotSupported {
            backend: backend_name(backend)
        }
    ))
}

pub(super) fn native_longform_options_from_cli(
    segment_mode: Option<NativeSegmentMode>,
    chunk_seconds: Option<f64>,
    segment_overlap_seconds: f64,
    vad_threshold_db: f32,
    vad_min_silence_ms: usize,
    vad_padding_ms: usize,
    min_segment_seconds: f64,
    suppress_silent_slices: bool,
) -> Result<openasr_core::LongFormOptions> {
    let mut options = openasr_core::LongFormOptions::default();
    if let Some(segment_mode) = segment_mode {
        options.mode = match segment_mode {
            NativeSegmentMode::Off => openasr_core::LongFormMode::Off,
            NativeSegmentMode::Auto => openasr_core::LongFormMode::Auto,
            NativeSegmentMode::Fixed => openasr_core::LongFormMode::Fixed,
            NativeSegmentMode::Energy => openasr_core::LongFormMode::Energy,
            NativeSegmentMode::Vad => openasr_core::LongFormMode::Vad,
        };
    }
    if let Some(chunk_seconds) = chunk_seconds {
        options.chunk_seconds = chunk_seconds as f32;
    }
    options.overlap_seconds = segment_overlap_seconds as f32;
    options.min_chunk_seconds = min_segment_seconds as f32;
    options.padding_seconds = vad_padding_ms as f32 / 1_000.0;
    options.energy_silence_threshold_db = vad_threshold_db;
    options.suppress_silent_slices = suppress_silent_slices;
    options.vad.min_silence_duration_ms =
        u32::try_from(vad_min_silence_ms).map_err(|_| anyhow::anyhow!(
            "--vad-min-silence-ms value {vad_min_silence_ms} is too large for native longform options"
        ))?;
    options.validate().map_err(|error| {
        anyhow::anyhow!("native longform options are invalid after CLI mapping: {error}")
    })?;
    Ok(options)
}

pub(super) fn native_longform_options_override_from_cli(
    cli: &NativeLongFormCliOptions,
) -> Result<Option<openasr_core::LongFormOptions>> {
    if *cli == NativeLongFormCliOptions::default() {
        return Ok(None);
    }
    native_longform_options_from_cli(
        cli.segment_mode,
        cli.chunk_seconds,
        cli.segment_overlap_seconds,
        cli.vad_threshold_db,
        cli.vad_min_silence_ms,
        cli.vad_padding_ms,
        cli.min_segment_seconds,
        cli.suppress_silent_slices,
    )
    .map(Some)
}

pub(super) fn backend_name(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Mock => "mock",
        BackendKind::Native => "native",
    }
}

pub(super) fn print_audio_input_notes(info: &AudioInputInfo) {
    for issue in &info.issues {
        match issue {
            AudioInputIssue::UnknownExtension(extension) => eprintln!(
                "Note: unrecognized audio extension \".{extension}\"; OpenASR will pass the file to the selected backend."
            ),
        }
    }
}

pub(super) fn print_audio_preparation_notes(prepared: &PreparedAudioInput) {
    if prepared.is_converted() {
        eprintln!(
            "Note: prepared {} as temporary 16 kHz mono PCM WAV for the selected backend.",
            prepared.original().path.display()
        );
    }
}

/// Renders `transcription` in every requested format and writes the output(s),
/// returning the paths written (empty when printed to stdout):
/// - one format, no `--output`, single input -> stdout;
/// - one format, `--output <file>`, single input -> that file;
/// - otherwise (several formats, or per-file batch mode) -> one
///   `<input_name>.<ext>` per format in `--output` (or next to the input).
pub(super) fn write_rendered_formats(
    transcription: &openasr_core::Transcription,
    formats: &[ResponseFormat],
    input: &Path,
    output: Option<&Path>,
    force_dir: bool,
) -> Result<Vec<PathBuf>> {
    if formats.len() <= 1 && !force_dir {
        let format = formats.first().copied().unwrap_or(ResponseFormat::Text);
        let rendered = render_transcription(transcription, format)
            .context("Could not render transcription output")?;
        return match output {
            Some(path) => {
                write_rendered_output_atomic(&rendered, path)?;
                Ok(vec![path.to_path_buf()])
            }
            None => {
                print!("{rendered}");
                Ok(Vec::new())
            }
        };
    }

    let dir = match output {
        Some(dir) => {
            ensure_batch_output_dir(dir)?;
            dir.to_path_buf()
        }
        None => input
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    };
    let mut written = Vec::with_capacity(formats.len());
    for format in formats {
        let rendered = render_transcription(transcription, *format)
            .context("Could not render transcription output")?;
        let path = batch_output_path(&dir, input, *format);
        write_rendered_output_atomic(&rendered, &path)?;
        written.push(path);
    }
    Ok(written)
}

pub(super) fn write_rendered_output(rendered: &str, output: Option<&Path>) -> Result<()> {
    let Some(output) = output else {
        print!("{rendered}");
        return Ok(());
    };

    write_rendered_output_atomic(rendered, output)?;

    Ok(())
}

pub(super) fn write_rendered_output_atomic(rendered: &str, output: &Path) -> Result<()> {
    atomic_write_text(output, rendered).map_err(|error| {
        if let Some(warning) = error.cleanup_warning() {
            eprintln!("{warning}");
        }
        anyhow::anyhow!("{error}")
    })
}

pub(super) fn parse_response_format(value: &str) -> Result<ResponseFormat, String> {
    ResponseFormat::from_str(value)
}

pub(super) fn parse_benchmark_format(value: &str) -> Result<BenchmarkFormat, String> {
    BenchmarkFormat::from_str(value)
}

pub(super) fn parse_backend_kind(value: &str) -> Result<BackendKind, String> {
    BackendKind::from_str(value)
}

pub(super) fn parse_transcription_task(value: &str) -> Result<TranscriptionTask, String> {
    TranscriptionTask::from_str(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header},
    };
    use std::ffi::{OsStr, OsString};
    use std::sync::{Mutex, MutexGuard, OnceLock};
    use tower::ServiceExt;

    struct EnvVarRestore {
        name: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarRestore {
        fn set(name: &'static str, value: &str) -> Self {
            Self::set_os(name, value)
        }

        fn set_os(name: &'static str, value: impl AsRef<OsStr>) -> Self {
            let previous = env::var_os(name);
            unsafe { env::set_var(name, value) };
            Self { name, previous }
        }

        fn remove(name: &'static str) -> Self {
            let previous = env::var_os(name);
            unsafe { env::remove_var(name) };
            Self { name, previous }
        }
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => unsafe { env::set_var(self.name, value) },
                None => unsafe { env::remove_var(self.name) },
            }
        }
    }

    fn env_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("env test lock poisoned")
    }

    fn with_env_lock<T>(run: impl FnOnce() -> T) -> T {
        let _guard = env_lock();
        run()
    }

    // Locks the three-tier priority `selected_model_ref` must keep: an explicit
    // `--model` always wins, then the persisted `config.default_model`, and only
    // with neither does the CLI fall back to `DEFAULT_MODEL_ID` -- the
    // bare-invocation convention that (post-refactor) is no longer implicitly
    // written into `config.json` (see `openasr_core::config::DEFAULT_MODEL_ID`
    // and `default_selection`).
    #[test]
    fn selected_model_ref_explicit_wins_over_config_default() {
        let config = OpenAsrConfig {
            default_model: Some("whisper-small".to_string()),
            ..OpenAsrConfig::default()
        };
        assert_eq!(
            selected_model_ref(Some("whisper-large-v3-turbo"), &config, &[]),
            "whisper-large-v3-turbo"
        );
    }

    #[test]
    fn selected_model_ref_falls_back_to_config_default_when_no_explicit_model() {
        let config = OpenAsrConfig {
            default_model: Some("whisper-small".to_string()),
            ..OpenAsrConfig::default()
        };
        assert_eq!(selected_model_ref(None, &config, &[]), "whisper-small");
    }

    #[test]
    fn selected_model_ref_falls_back_to_default_model_id_when_config_default_is_unset() {
        // A fresh config (or one built by `OpenAsrConfig::default()`) has
        // `default_model: None` -- the CLI convention fallback, not a config value,
        // must still resolve to something usable.
        let config = OpenAsrConfig::default();
        assert_eq!(config.default_model, None);
        assert_eq!(selected_model_ref(None, &config, &[]), DEFAULT_MODEL_ID);
    }

    #[test]
    fn native_longform_options_maps_energy_mode() {
        let options = native_longform_options_from_cli(
            Some(NativeSegmentMode::Energy),
            Some(42.0),
            0.75,
            -32.0,
            300,
            200,
            1.5,
            true,
        )
        .expect("options");
        assert_eq!(options.mode, openasr_core::LongFormMode::Energy);
        assert_eq!(options.chunk_seconds, 42.0);
        assert_eq!(options.overlap_seconds, 0.75);
        assert_eq!(options.min_chunk_seconds, 1.5);
        assert_eq!(options.padding_seconds, 0.2);
        assert_eq!(options.energy_silence_threshold_db, -32.0);
        assert!(options.suppress_silent_slices);
        assert_eq!(options.vad.min_silence_duration_ms, 300);
    }

    #[test]
    fn native_longform_options_fails_closed_on_invalid_overlap() {
        let error = native_longform_options_from_cli(
            Some(NativeSegmentMode::Fixed),
            Some(2.0),
            2.0,
            -38.0,
            450,
            250,
            1.0,
            false,
        )
        .expect_err("must fail");
        assert!(
            error
                .to_string()
                .contains("native longform options are invalid")
        );
    }

    #[test]
    fn native_longform_options_override_omits_default_cli_values() {
        let options =
            native_longform_options_override_from_cli(&NativeLongFormCliOptions::default())
                .expect("options");
        assert!(options.is_none());
    }

    #[test]
    fn native_longform_cli_defaults_match_core_defaults() {
        let cli = NativeLongFormCliOptions::default();
        let mapped = native_longform_options_from_cli(
            cli.segment_mode,
            cli.chunk_seconds,
            cli.segment_overlap_seconds,
            cli.vad_threshold_db,
            cli.vad_min_silence_ms,
            cli.vad_padding_ms,
            cli.min_segment_seconds,
            cli.suppress_silent_slices,
        )
        .expect("mapped defaults");
        assert_eq!(mapped, openasr_core::LongFormOptions::default());
    }

    #[test]
    fn native_longform_options_override_keeps_explicit_non_default_values() {
        let options = native_longform_options_override_from_cli(&NativeLongFormCliOptions {
            segment_mode: Some(NativeSegmentMode::Energy),
            suppress_silent_slices: true,
            ..NativeLongFormCliOptions::default()
        })
        .expect("options");
        let options = options.expect("override");
        assert_eq!(options.mode, openasr_core::LongFormMode::Energy);
        assert!(options.suppress_silent_slices);
    }

    #[test]
    fn native_longform_options_override_keeps_explicit_auto_mode() {
        let options = native_longform_options_override_from_cli(&NativeLongFormCliOptions {
            segment_mode: Some(NativeSegmentMode::Auto),
            ..NativeLongFormCliOptions::default()
        })
        .expect("options");
        let options = options.expect("override");
        assert_eq!(options.mode, openasr_core::LongFormMode::Auto);
    }

    #[test]
    fn phrase_bias_cli_options_map_repeated_hotwords_to_core_config() {
        let config = phrase_bias_options_from_cli(&PhraseBiasCliOptions {
            hotwords: vec![" OpenASR  Core ".to_string(), "Qwen".to_string()],
            hotword_boost: Some(3.5),
        })
        .expect("phrase bias options")
        .expect("phrase bias config");

        assert_eq!(config.entries().len(), 2);
        assert_eq!(config.entries()[0].phrase(), "OpenASR Core");
        assert_eq!(config.entries()[0].boost(), 3.5);
        assert_eq!(config.entries()[1].phrase(), "Qwen");
    }

    #[test]
    fn phrase_bias_cli_options_use_default_boost_for_hotword() {
        let config = phrase_bias_options_from_cli(&PhraseBiasCliOptions {
            hotwords: vec!["OpenASR".to_string()],
            hotword_boost: None,
        })
        .expect("phrase bias options")
        .expect("phrase bias config");

        assert_eq!(
            config.entries()[0].boost(),
            openasr_core::DEFAULT_PHRASE_BIAS_BOOST
        );
    }

    #[test]
    fn phrase_bias_cli_options_reject_boost_without_hotword() {
        let error = phrase_bias_options_from_cli(&PhraseBiasCliOptions {
            hotwords: Vec::new(),
            hotword_boost: Some(2.0),
        })
        .expect_err("boost without hotword must fail")
        .to_string();

        assert!(error.contains("--hotword-boost requires at least one --hotword"));
    }

    #[test]
    fn phrase_bias_cli_options_do_not_echo_invalid_phrase() {
        let error = phrase_bias_options_from_cli(&PhraseBiasCliOptions {
            hotwords: vec![" \t\n ".to_string()],
            hotword_boost: Some(2.0),
        })
        .expect_err("empty hotword must fail")
        .to_string();

        assert!(error.contains("Invalid phrase bias CLI options"));
        assert!(!error.contains(" \t\n "));
    }

    #[test]
    fn phrase_bias_cli_uses_backend_capabilities() {
        let config = openasr_core::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)])
            .expect("phrase bias fixture");

        ensure_phrase_bias_supported(BackendKind::Native, None, Some(&config))
            .expect("native backend advertises phrase-bias support");

        let error = ensure_phrase_bias_supported(BackendKind::Mock, None, Some(&config))
            .expect_err("mock phrase bias should fail closed")
            .to_string();
        assert!(error.contains("Phrase bias / hotword boosting is not supported"));
        assert!(error.contains(backend_name(BackendKind::Mock)));
        assert!(error.contains("silently ignoring phrase_bias"));
    }

    #[test]
    fn phrase_bias_cli_rejects_xasr_model_pack_early() {
        let temp = tempfile::tempdir().unwrap();
        let pack_path = temp.path().join("xasr-cli.oasr");
        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("openasr.model.id".to_string(), "xasr-cli".to_string());
        metadata.insert(
            openasr_core::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            openasr_core::models::oasr_metadata::OASR_PACKAGE_VERSION_V1.to_string(),
        );
        metadata.insert(
            openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
            "xasr-zipformer".to_string(),
        );
        metadata.insert(
            openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
            openasr_core::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID.to_string(),
        );
        metadata.insert(
            openasr_core::models::oasr_metadata::OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
            openasr_core::XASR_ZIPFORMER_AUDIO_FRONTEND_ID.to_string(),
        );
        metadata.insert(
            openasr_core::models::oasr_metadata::OASR_METADATA_KEY_DECODE_POLICY.to_string(),
            openasr_core::XASR_ZIPFORMER_DECODE_POLICY_ID.to_string(),
        );
        metadata.insert(
            openasr_core::GGML_TOKENIZER_ID_KEY.to_string(),
            openasr_core::XASR_ZIPFORMER_TOKENIZER_ID.to_string(),
        );
        let spec = openasr_core::testing::TinyGgufFixtureSpec::new(metadata);
        openasr_core::testing::write_tiny_gguf_runtime_source(&pack_path, &spec).unwrap();
        let config = openasr_core::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)])
            .expect("phrase bias fixture");

        let error =
            ensure_phrase_bias_supported(BackendKind::Native, Some(&pack_path), Some(&config))
                .expect_err("xasr phrase bias should fail early")
                .to_string();

        assert!(error.contains("--hotword is not supported"));
        assert!(error.contains("xasr-zipformer"));
        assert!(error.contains("silently ignoring phrase_bias"));
    }

    #[test]
    fn diarization_cli_uses_backend_capabilities() {
        let _guard = env_lock();
        let temp = tempfile::tempdir().unwrap();
        // Isolate the model-agnostic VAD + speaker-embedder probe from the host
        // machine's installed packs so the fail-closed expectations are hermetic.
        let _campplus_pack = EnvVarRestore::remove("OPENASR_CAMPPLUS_PACK");
        let _wespeaker_pack = EnvVarRestore::remove("OPENASR_WESPEAKER_PACK");
        let _speaker_embedder = EnvVarRestore::remove("OPENASR_SPEAKER_EMBEDDER");
        let _home = EnvVarRestore::set_os("OPENASR_HOME", temp.path());

        let error = ensure_diarization_supported(BackendKind::Mock, None, true)
            .expect_err("mock diarization should fail closed")
            .to_string();
        assert!(error.contains("speaker-embedder pack"));
        assert!(error.contains(backend_name(BackendKind::Mock)));

        let base_runtime_path = temp.path().join("cohere-base.oasr");
        let base_spec =
            openasr_core::testing::TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-base");
        openasr_core::testing::write_tiny_gguf_runtime_source(&base_runtime_path, &base_spec)
            .unwrap();

        let error =
            ensure_diarization_supported(BackendKind::Native, Some(&base_runtime_path), true)
                .expect_err("base native pack must keep diarization fail-closed")
                .to_string();
        assert!(error.contains("speaker-embedder pack"));
        assert!(error.contains(backend_name(BackendKind::Native)));

        // Installing the WeSpeaker embedder pack enables the model-agnostic
        // path for any native pack, and for the no-pack-path live preflight.
        let wespeaker_pack = temp.path().join("wespeaker.oasr");
        std::fs::write(&wespeaker_pack, b"GGUF\x00\x00\x00\x00").unwrap();
        let _installed_wespeaker_pack =
            EnvVarRestore::set_os("OPENASR_WESPEAKER_PACK", &wespeaker_pack);
        ensure_diarization_supported(BackendKind::Native, Some(&base_runtime_path), true)
            .expect("WeSpeaker pack should pass the CLI gate for any native pack");
        ensure_diarization_supported(BackendKind::Native, None, true)
            .expect("WeSpeaker pack should pass the CLI gate without a pack path");

        let declared_runtime_path = temp.path().join("cohere-diarize.oasr");
        let declared_spec =
            openasr_core::testing::TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready(
                "cohere-diarize",
            )
            .with_metadata(
                openasr_core::models::oasr_metadata::OASR_METADATA_KEY_FEATURE_DIARIZATION,
                openasr_core::models::oasr_metadata::OASR_FEATURE_DIARIZATION_COHERE_TOKEN_STREAM_V1,
            )
            .with_string_array_metadata("tokenizer.ggml.tokens", cohere_diarization_tokens());
        openasr_core::testing::write_tiny_gguf_runtime_source(
            &declared_runtime_path,
            &declared_spec,
        )
        .unwrap();

        ensure_diarization_supported(BackendKind::Native, Some(&declared_runtime_path), true)
            .expect("declared Cohere diarization pack should pass the CLI gate");
    }

    #[test]
    fn word_timestamps_alignment_supported_only_when_aligned_requested() {
        // Absent / approximate never gates on backend -- only `aligned` does.
        ensure_word_timestamps_alignment_supported(BackendKind::Mock, None)
            .expect("no word-timestamps request is always fine");
        ensure_word_timestamps_alignment_supported(
            BackendKind::Mock,
            Some(WordTimestampsMode::Approximate),
        )
        .expect("approximate word timestamps do not require the native backend");
    }

    #[test]
    fn word_timestamps_alignment_requires_native_backend() {
        let error = ensure_word_timestamps_alignment_supported(
            BackendKind::Mock,
            Some(WordTimestampsMode::Aligned),
        )
        .expect_err("aligned refinement should reject the mock backend")
        .to_string();
        assert!(error.contains("--word-timestamps=aligned"));
        assert!(error.contains("native"));

        ensure_word_timestamps_alignment_supported(
            BackendKind::Native,
            Some(WordTimestampsMode::Aligned),
        )
        .expect("aligned refinement is allowed on the native backend");
    }

    #[test]
    fn word_timestamps_pack_install_is_a_no_op_without_aligned_mode() {
        let _guard = env_lock();
        let temp = tempfile::tempdir().unwrap();
        let _home = EnvVarRestore::set_os("OPENASR_HOME", temp.path());

        // Neither absent nor approximate ever touches the catalog/network.
        ensure_cli_word_timestamps_pack_installed(BackendKind::Native, None)
            .expect("no word-timestamps request never installs a pack");
        ensure_cli_word_timestamps_pack_installed(
            BackendKind::Native,
            Some(WordTimestampsMode::Approximate),
        )
        .expect("approximate word timestamps never install the forced-aligner pack");
        ensure_cli_word_timestamps_pack_installed(
            BackendKind::Mock,
            Some(WordTimestampsMode::Aligned),
        )
        .expect("the mock backend never needs the native-only forced-aligner pack");
    }

    fn cohere_diarization_tokens() -> [&'static str; 31] {
        [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|en|>",
            "<|pnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|timestamp|>",
            "<|nodiarize|>",
            "<|diarize|>",
            "<|endoftext|>",
            "<|spltoken0|>",
            "▁fixture11",
            "▁fixture12",
            "▁fixture13",
            "▁fixture14",
            "▁fixture15",
            "▁fixture16",
            "▁fixture17",
            "▁fixture18",
            "▁fixture19",
            "▁fixture20",
            "▁fixture21",
            "▁fixture22",
            "▁fixture23",
            "▁fixture24",
            "▁fixture25",
            "▁fixture26",
            "▁fixture27",
            "▁fixture28",
            "▁fixture29",
        ]
    }

    #[test]
    fn remote_serve_tls_sans_include_bound_ip_and_localhost() {
        let names = default_tls_subject_alt_names(
            "127.0.0.1:8443".parse().unwrap(),
            &["OpenASR.local".to_string(), " ".to_string()],
        );

        assert_eq!(
            names,
            vec![
                "OpenASR.local".to_string(),
                "127.0.0.1".to_string(),
                "localhost".to_string()
            ]
        );
    }

    #[test]
    fn remote_serve_tls_sans_do_not_add_unspecified_address() {
        let names = default_tls_subject_alt_names("0.0.0.0:8443".parse().unwrap(), &[]);

        assert!(names.is_empty());
    }

    #[test]
    fn remote_serve_pairing_token_env_must_exist_and_be_nonempty() {
        with_env_lock(|| {
            unsafe { env::remove_var("OPENASR_TEST_PAIRING_TOKEN") };
            let error = serve_launch_options(
                "127.0.0.1:8443".parse().unwrap(),
                ServeSecurityOptions {
                    pairing_admin_token_env: Some("OPENASR_TEST_PAIRING_TOKEN".to_string()),
                    ..Default::default()
                },
                Vec::new(),
            )
            .expect_err("missing env must fail")
            .to_string();
            assert!(error.contains("OPENASR_TEST_PAIRING_TOKEN"));

            unsafe { env::set_var("OPENASR_TEST_PAIRING_TOKEN", "  ") };
            let error = serve_launch_options(
                "127.0.0.1:8443".parse().unwrap(),
                ServeSecurityOptions {
                    pairing_admin_token_env: Some("OPENASR_TEST_PAIRING_TOKEN".to_string()),
                    ..Default::default()
                },
                Vec::new(),
            )
            .expect_err("empty env must fail")
            .to_string();
            assert!(error.contains("must not be empty"));
            unsafe { env::remove_var("OPENASR_TEST_PAIRING_TOKEN") };
        });
    }

    #[tokio::test]
    async fn remote_serve_pairing_token_env_configures_real_pairing_auth() {
        let launch_options = {
            let _guard = env_lock();
            let _restore = EnvVarRestore::set("OPENASR_TEST_PAIRING_TOKEN_OK", "pair-admin-secret");
            serve_launch_options(
                "127.0.0.1:8443".parse().unwrap(),
                ServeSecurityOptions {
                    tls_self_signed: true,
                    pairing_admin_token_env: Some("OPENASR_TEST_PAIRING_TOKEN_OK".to_string()),
                    ..Default::default()
                },
                Vec::new(),
            )
            .expect("serve launch options")
        };

        match &launch_options.tls {
            openasr_server::ServerTlsConfig::SelfSigned { subject_alt_names } => {
                assert!(subject_alt_names.iter().any(|name| name == "127.0.0.1"));
                assert!(subject_alt_names.iter().any(|name| name == "localhost"));
            }
            openasr_server::ServerTlsConfig::Disabled => panic!("expected self-signed TLS"),
        }

        let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
            openasr_server::ServerRuntime::default(),
            openasr_server::DistributionRuntime::default(),
            launch_options,
        );
        let create = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/pairing/requests")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"device_name":"CLI Remote"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(create.status(), StatusCode::ACCEPTED);
        let create_body = to_bytes(create.into_body(), 1024 * 64).await.unwrap();
        let create_json: serde_json::Value = serde_json::from_slice(&create_body).unwrap();
        let request_id = create_json["request_id"].as_str().unwrap();

        let unauthorized = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/pairing/requests/{request_id}/approve"))
                    .header(header::AUTHORIZATION, "Bearer wrong-token")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let approved = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/pairing/requests/{request_id}/approve"))
                    .header(header::AUTHORIZATION, "Bearer pair-admin-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(approved.status(), StatusCode::OK);
    }

    #[test]
    fn supervised_daemon_launch_detects_instance_token_env() {
        with_env_lock(|| {
            let _removed = EnvVarRestore::remove("OPENASR_SERVER_INSTANCE_TOKEN");
            assert!(!supervised_daemon_launch());
            let _blank = EnvVarRestore::set("OPENASR_SERVER_INSTANCE_TOKEN", "  ");
            assert!(!supervised_daemon_launch());
            let _set = EnvVarRestore::set("OPENASR_SERVER_INSTANCE_TOKEN", "desktop-token");
            assert!(supervised_daemon_launch());
        });
    }

    async fn models_status(app: axum::Router, bearer: Option<&str>) -> StatusCode {
        let mut request = Request::builder().method("GET").uri("/v1/models");
        if let Some(bearer) = bearer {
            request = request.header(header::AUTHORIZATION, format!("Bearer {bearer}"));
        }
        app.oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn serve_loopback_without_configured_keys_leaves_auth_disabled() {
        let launch_options = serve_launch_options(
            "127.0.0.1:8080".parse().unwrap(),
            ServeSecurityOptions::default(),
            Vec::new(),
        )
        .expect("serve launch options");
        let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
            openasr_server::ServerRuntime::default(),
            openasr_server::DistributionRuntime::default(),
            launch_options,
        );

        assert_eq!(models_status(app, None).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn serve_loopback_with_configured_key_requires_matching_bearer() {
        let key_hash = openasr_core::apikeys::hash_api_key_token("oasr_sk_test-agent-key");
        let launch_options = serve_launch_options(
            "127.0.0.1:8080".parse().unwrap(),
            ServeSecurityOptions::default(),
            vec![key_hash],
        )
        .expect("serve launch options");
        let build_app = || {
            openasr_server::app_with_runtime_and_distribution_and_launch_options(
                openasr_server::ServerRuntime::default(),
                openasr_server::DistributionRuntime::default(),
                launch_options.clone(),
            )
        };

        assert_eq!(
            models_status(build_app(), None).await,
            StatusCode::UNAUTHORIZED,
            "loopback must require the key once one is configured"
        );
        assert_eq!(
            models_status(build_app(), Some("wrong-key")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            models_status(build_app(), Some("oasr_sk_test-agent-key")).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn serve_non_loopback_ignores_configured_keys_without_pairing() {
        let key_hash = openasr_core::apikeys::hash_api_key_token("oasr_sk_test-agent-key");
        let launch_options = serve_launch_options(
            "0.0.0.0:8080".parse().unwrap(),
            ServeSecurityOptions::default(),
            vec![key_hash],
        )
        .expect("serve launch options");
        let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
            openasr_server::ServerRuntime::default(),
            openasr_server::DistributionRuntime::default(),
            launch_options,
        );

        // A locally-created API key must never substitute for device pairing
        // on a non-loopback bind: `validate_listen_security` is what actually
        // fail-closes this bind (no TLS/auth), but at the auth-construction
        // level the key must not have been wired in either.
        assert_eq!(
            models_status(app, Some("oasr_sk_test-agent-key")).await,
            StatusCode::OK,
            "non-loopback must not honor a loopback-only API key"
        );
    }
}
