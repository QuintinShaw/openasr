use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender, TrySendError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use cpal::{
    Device, Host, SampleFormat, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use openasr_core::{
    AudioPreparationOptions, BufferedUtterance, RealtimeAudioFormat, RealtimeAudioFrame,
    RealtimeBufferConfig, RealtimeErrorCode, RealtimeErrorEvent, RealtimeEventEnvelope,
    RealtimeExportFormat, RealtimeLifecycleAction, RealtimePostProcessor, RealtimeSessionConfig,
    RealtimeSessionController, RealtimeTranscriptEvent, RealtimeTranscriptHistory,
    RealtimeUtteranceEndReason, RealtimeVadEvent, SpeechBoundaryEvent, TranscriptLifecycleResult,
    TranscriptSegmentId, TranscriptUpdate, TranscriptUtteranceId, VadConfig, VadMode,
    atomic_write_text, prepare_audio_input,
};

use crate::catalog_cli::load_cli_model_catalog;
use crate::{
    BackendKind, RuntimePathOverrides, TranscriptionRequest, backend_name,
    ensure_cli_diarization_packs_installed, ensure_diarization_supported, find_model, load_config,
    openasr_home, resolve_backend, runtime_registry, selected_model_ref, transcribe_with_backend,
    validate_local_native_model_pack_path,
};

const LIVE_CAPTURE_QUEUE_CAPACITY: usize = 64;
const LIVE_TRANSCRIPTION_QUEUE_CAPACITY: usize = 4;
const LIVE_SESSION_ID: &str = "rt_cli_live";
const LIVE_TEMP_PREFIX: &str = "openasr-live-utterance-";
const DEFAULT_LIVE_MAX_BUFFERED_FRAMES: usize = 1_510;
const DEFAULT_OBS_MAX_LINES: usize = 3;
const DEFAULT_STREAMING_PARTIAL_INTERVAL_MS: u64 = 300;
const DEFAULT_STREAMING_PARTIAL_WINDOW_MS: u32 = 3_000;
const PARTIAL_ROLLBACK_SUPPRESS_MIN_DELTA_CHARS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveSource {
    Mic,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LiveOutputFormat {
    Text,
    Jsonl,
}

#[derive(Debug, Clone)]
pub(crate) struct LiveCommandOptions<'a> {
    pub source: LiveSource,
    pub list_devices: bool,
    pub device: Option<String>,
    pub input_file: Option<PathBuf>,
    pub model: Option<&'a str>,
    pub backend: Option<BackendKind>,
    pub model_pack: Option<&'a Path>,
    pub output_format: LiveOutputFormat,
    pub max_seconds: Option<u64>,
    pub max_utterances: Option<usize>,
    pub frame_duration_ms: u32,
    pub speech_start_ms: Option<u32>,
    pub speech_stop_ms: Option<u32>,
    pub pre_roll_ms: Option<u32>,
    pub max_utterance_ms: Option<u32>,
    pub no_speech_timeout_ms: Option<u32>,
    pub energy_threshold: Option<f32>,
    pub partial_interval_ms: Option<u64>,
    pub partial_window_ms: Option<u32>,
    pub diarize: bool,
    pub save_path: Option<PathBuf>,
    pub save_join_segments: bool,
    pub save_suggest_title: bool,
    pub obs_text_file: Option<PathBuf>,
    pub obs_max_lines: Option<usize>,
    pub obs_clear_on_start: bool,
    pub obs_clear_on_stop: bool,
    pub markdown_note_path: Option<PathBuf>,
    pub markdown_append: bool,
    pub markdown_title: Option<String>,
    pub markdown_suggest_title: bool,
    pub runtime_paths: RuntimePathOverrides,
    pub consent: crate::consent::PullConsent,
}

pub(crate) fn parse_live_source(value: &str) -> std::result::Result<LiveSource, String> {
    match value {
        "mic" => Ok(LiveSource::Mic),
        "system" => Ok(LiveSource::System),
        other => Err(format!(
            "Unsupported live source '{other}'. Use one of: mic, system."
        )),
    }
}

pub(crate) fn parse_live_output_format(
    value: &str,
) -> std::result::Result<LiveOutputFormat, String> {
    match value {
        "text" => Ok(LiveOutputFormat::Text),
        "jsonl" => Ok(LiveOutputFormat::Jsonl),
        other => Err(format!(
            "Unsupported live output format '{other}'. Use one of: text, jsonl."
        )),
    }
}

pub(crate) async fn run_live(options: LiveCommandOptions<'_>) -> Result<()> {
    validate_live_limits(options.max_seconds, options.max_utterances)?;

    let host = cpal::default_host();
    if options.list_devices {
        match options.source {
            LiveSource::Mic => list_input_devices(&host)?,
            LiveSource::System => list_system_audio_source(),
        }
        return Ok(());
    }

    let home = openasr_home()?;
    let config = load_config(&home)?;
    let catalog = load_cli_model_catalog(&home)?;
    let cards = runtime_registry(catalog.as_ref()).context("Could not load model registry")?;
    let model_ref = selected_model_ref(options.model, &config, &cards);
    let card = find_model(&cards, &model_ref)?.card;
    let backend = resolve_backend(options.backend, &config)?;
    // CLI-only consent-pull: native without an explicit --model-pack ensures the
    // resolved model is installed first (see the transcribe handler).
    if backend == BackendKind::Native && options.model_pack.is_none() {
        crate::pull_cli::ensure_asr_model_installed(options.model, &config, &options.consent)?;
    }
    let model_pack_path = resolve_live_model_pack(
        backend,
        options.model,
        &config,
        options.model_pack,
        catalog.as_ref(),
    )?;
    ensure_cli_diarization_packs_installed(backend, model_pack_path.as_deref(), options.diarize)?;
    ensure_diarization_supported(backend, model_pack_path.as_deref(), options.diarize)?;
    ensure_live_backend_ready(backend, model_pack_path.as_deref())?;
    let live_config =
        LivePipelineConfig::from_options(&options, card.id.clone(), model_pack_path.clone())?;
    let save_options = build_save_options(&options)?;
    let obs_options = build_obs_options(&options)?;
    let markdown_note_options =
        build_markdown_note_options(&options, backend, &card.id, timestamp_now())?;
    validate_no_sink_path_collisions(
        save_options.as_ref(),
        obs_options.as_ref(),
        markdown_note_options.as_ref(),
    )?;

    if let Some(input_file) = options.input_file.as_deref() {
        return run_live_from_input_file(
            input_file,
            &options,
            backend,
            model_pack_path,
            live_config,
            save_options,
            obs_options,
            markdown_note_options,
        );
    }

    if options.source == LiveSource::System {
        return run_live_from_system_audio(
            &options,
            backend,
            model_pack_path,
            live_config,
            save_options,
            obs_options,
            markdown_note_options,
        );
    }

    let device = select_input_device(&host, options.device.as_deref())?;
    let device_label = device_label(&device);
    let supported_config = device
        .default_input_config()
        .with_context(|| format!("Could not query default input config for {device_label}"))?;
    let sample_format = supported_config.sample_format();
    let stream_config: StreamConfig = supported_config.clone().into();
    let normalizer = LiveAudioNormalizer::new(
        stream_config.sample_rate,
        stream_config.channels,
        options.frame_duration_ms,
    )?;
    let (sender, receiver) = mpsc::sync_channel(LIVE_CAPTURE_QUEUE_CAPACITY);
    let overflowed = Arc::new(AtomicBool::new(false));
    let stream = build_input_stream(&device, &stream_config, sample_format, sender, &overflowed)?;
    stream
        .play()
        .with_context(|| format!("Could not start microphone stream for {device_label}"))?;

    let stop_requested = Arc::new(AtomicBool::new(false));
    let signal_flag = Arc::clone(&stop_requested);
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_flag.store(true, Ordering::SeqCst);
        }
    });

    let mut pipeline = LivePipeline::new(
        live_config,
        options.output_format,
        options.max_utterances,
        save_options,
    )?;
    pipeline.configure_prototype_sinks(obs_options, markdown_note_options);
    pipeline.start()?;
    eprintln!(
        "OpenASR live microphone started on {device_label} ({sample_format}, {} Hz, {} channel(s)). Streaming partial/final updates are enabled.",
        stream_config.sample_rate, stream_config.channels
    );
    let transcription_worker = LiveTranscriptionWorker::spawn(backend, model_pack_path);
    let mut capture = LiveCaptureRun {
        receiver,
        normalizer,
        overflowed,
        started_at: Instant::now(),
        max_seconds: options.max_seconds,
        max_utterances: options.max_utterances,
        stop_requested,
        transcription_worker,
    };
    let result = capture.run(&mut pipeline);
    drop(stream);
    result
}

fn run_live_from_system_audio(
    options: &LiveCommandOptions<'_>,
    backend: BackendKind,
    model_pack_path: Option<PathBuf>,
    live_config: LivePipelineConfig,
    save_options: Option<LiveSaveOptions>,
    obs_options: Option<LiveObsOptions>,
    markdown_note_options: Option<LiveMarkdownNoteOptions>,
) -> Result<()> {
    let support = openasr_system_audio::support_status();
    if !support.supported {
        bail!("System audio capture is not available: {}", support.detail);
    }

    let normalizer = LiveAudioNormalizer::new(
        openasr_core::DEFAULT_REALTIME_SAMPLE_RATE_HZ,
        1,
        options.frame_duration_ms,
    )?;
    let (sender, receiver) = mpsc::sync_channel(LIVE_CAPTURE_QUEUE_CAPACITY);
    let overflowed = Arc::new(AtomicBool::new(false));
    let capture_stop = Arc::new(AtomicBool::new(false));
    let stop_requested = Arc::new(AtomicBool::new(false));
    let signal_stop_requested = Arc::clone(&stop_requested);
    let signal_capture_stop = Arc::clone(&capture_stop);
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            signal_stop_requested.store(true, Ordering::SeqCst);
            signal_capture_stop.store(true, Ordering::SeqCst);
        }
    });

    let capture_stop_worker = Arc::clone(&capture_stop);
    let overflow_flag = Arc::clone(&overflowed);
    let (capture_result_tx, capture_result_rx) =
        mpsc::channel::<Result<String, openasr_system_audio::CaptureBackendError>>();
    let capture_worker = thread::spawn(move || {
        let result = openasr_system_audio::run_loopback_capture(
            capture_stop_worker,
            |samples| {
                if samples.is_empty() {
                    return Ok(());
                }
                match sender.try_send(CaptureChunk::I16(samples)) {
                    Ok(()) => Ok(()),
                    Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                        overflow_flag.store(true, Ordering::SeqCst);
                        Ok(())
                    }
                }
            },
            |message| {
                eprintln!("OpenASR live system audio: {message}");
                Ok(())
            },
        );
        let _ = capture_result_tx.send(result);
    });

    let mut pipeline = LivePipeline::new(
        live_config,
        options.output_format,
        options.max_utterances,
        save_options,
    )?;
    pipeline.configure_prototype_sinks(obs_options, markdown_note_options);
    pipeline.start()?;
    eprintln!(
        "OpenASR live system audio started with {} on {}. Streaming partial/final updates are enabled.",
        support.label, support.platform
    );
    let transcription_worker = LiveTranscriptionWorker::spawn(backend, model_pack_path);
    let mut capture = LiveCaptureRun {
        receiver,
        normalizer,
        overflowed,
        started_at: Instant::now(),
        max_seconds: options.max_seconds,
        max_utterances: options.max_utterances,
        stop_requested,
        transcription_worker,
    };
    let pipeline_result = capture.run(&mut pipeline);
    capture_stop.store(true, Ordering::SeqCst);
    let _ = capture_worker.join();

    match (pipeline_result, capture_result_rx.try_recv().ok()) {
        (Err(error), _) => Err(error),
        (Ok(()), Some(Err(error))) => {
            Err(anyhow::anyhow!("{} ({})", error.message, error.diagnostic))
        }
        (Ok(()), _) => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_live_from_input_file(
    input_file: &Path,
    options: &LiveCommandOptions<'_>,
    backend: BackendKind,
    model_pack_path: Option<PathBuf>,
    live_config: LivePipelineConfig,
    save_options: Option<LiveSaveOptions>,
    obs_options: Option<LiveObsOptions>,
    markdown_note_options: Option<LiveMarkdownNoteOptions>,
) -> Result<()> {
    let mut pipeline = LivePipeline::new(
        live_config,
        options.output_format,
        options.max_utterances,
        save_options,
    )?;
    pipeline.configure_prototype_sinks(obs_options, markdown_note_options);
    pipeline.start()?;
    eprintln!(
        "OpenASR live file simulation started for {} (frame={}ms).",
        input_file.display(),
        options.frame_duration_ms
    );
    let mut transcription_worker = LiveTranscriptionWorker::spawn(backend, model_pack_path);
    let samples = prepare_input_file_samples_pcm16_mono_16khz(input_file, options)?;
    let frame_sample_count = RealtimeAudioFormat::pcm16_mono_16khz()
        .sample_count_for_duration_ms(options.frame_duration_ms)?;
    let mut start_ms = 0_u64;
    for (frame_seq, chunk) in samples.chunks(frame_sample_count).enumerate() {
        let frame = RealtimeAudioFrame::new(
            frame_seq as u64,
            start_ms,
            RealtimeAudioFormat::pcm16_mono_16khz(),
            chunk.to_vec(),
        )?;
        pipeline.process_frame(frame, &mut transcription_worker)?;
        pipeline.drain_finished_transcriptions(&mut transcription_worker)?;
        start_ms += options.frame_duration_ms as u64;
        if options
            .max_utterances
            .is_some_and(|limit| pipeline.accepted_utterances >= limit)
        {
            break;
        }
    }
    pipeline.shutdown(start_ms, &mut transcription_worker, true)
}

fn prepare_input_file_samples_pcm16_mono_16khz(
    input_file: &Path,
    options: &LiveCommandOptions<'_>,
) -> Result<Vec<i16>> {
    let prepared = prepare_audio_input(
        input_file,
        &AudioPreparationOptions::new(BackendKind::Native)
            .with_ffmpeg_bin(options.runtime_paths.ffmpeg_bin.clone())
            .with_ffmpeg_bin_explicit(options.runtime_paths.ffmpeg_bin.is_some())
            .with_native_non_wav_conversion(true),
    )?;
    let mut reader = hound::WavReader::open(prepared.path()).with_context(|| {
        format!(
            "Could not open prepared WAV audio at {}",
            prepared.path().display()
        )
    })?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        bail!(
            "Prepared live input WAV must be mono 16kHz, got channels={}, sample_rate={}, bits_per_sample={}, sample_format={:?}.",
            spec.channels,
            spec.sample_rate,
            spec.bits_per_sample,
            spec.sample_format
        );
    }
    let mut samples = Vec::with_capacity(reader.duration() as usize);
    if spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample == 16 {
        for sample in reader.samples::<i16>() {
            samples.push(sample.context("Could not decode PCM16 sample from prepared WAV input")?);
        }
        return Ok(samples);
    }
    if spec.sample_format == hound::SampleFormat::Float && spec.bits_per_sample == 32 {
        for sample in reader.samples::<f32>() {
            let value =
                sample.context("Could not decode float32 sample from prepared WAV input")?;
            let scaled = (value * i16::MAX as f32).round();
            samples.push(scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16);
        }
        return Ok(samples);
    }
    bail!(
        "Prepared live input WAV sample format is unsupported: sample_format={:?}, bits_per_sample={}.",
        spec.sample_format,
        spec.bits_per_sample
    )
}

fn ensure_live_backend_ready(backend: BackendKind, model_pack: Option<&Path>) -> Result<()> {
    if backend == BackendKind::Native && model_pack.is_none() {
        return Err(anyhow::anyhow!(openasr_core::BackendError::NativeFailClosed {
            reason:
                "live native transcription requires --model-pack with a local .oasr runtime pack file".to_string(),
        }));
    }
    Ok(())
}

fn resolve_live_model_pack(
    backend: BackendKind,
    model: Option<&str>,
    config: &openasr_core::OpenAsrConfig,
    model_pack: Option<&Path>,
    catalog: Option<&openasr_core::ModelCatalog>,
) -> Result<Option<PathBuf>> {
    if backend != BackendKind::Native {
        if model_pack.is_some() {
            bail!("--model-pack is only supported with --backend native for live mode.");
        }
        return Ok(None);
    }
    match model_pack {
        Some(path) => {
            let validated = validate_local_native_model_pack_path(path).map_err(|error| {
                anyhow::anyhow!(
                    "Native model-pack path rejected for live mode: {error}\nNative live execution is local-path-only and fail-closed."
                )
            })?;
            Ok(Some(validated))
        }
        // No explicit pack: resolve an installed pack by model id. The consent-pull
        // in run_live already ensured it is installed; this never pulls. Forward
        // the catalog already loaded by run_live so family:tag aliases (e.g.
        // `qwen:q8`) resolve the same way here as they do for `transcribe`.
        None => Ok(Some(
            crate::native_segment_cli::resolve_installed_native_pack(model, config, catalog)?,
        )),
    }
}

fn build_save_options(options: &LiveCommandOptions<'_>) -> Result<Option<LiveSaveOptions>> {
    let Some(path) = options.save_path.clone() else {
        return Ok(None);
    };
    let format = RealtimeExportFormat::from_extension(&path).ok_or_else(|| {
        anyhow::anyhow!(
            "Unsupported live export extension for '{}'. Use one of: .txt, .json, .md, .srt, .vtt.",
            path.display()
        )
    })?;
    validate_save_target_path(&path)?;
    Ok(Some(LiveSaveOptions {
        path,
        format,
        post_processor: RealtimePostProcessor {
            join_segments: options.save_join_segments,
            suggest_title: options.save_suggest_title,
            ..RealtimePostProcessor::default()
        },
    }))
}

fn build_obs_options(options: &LiveCommandOptions<'_>) -> Result<Option<LiveObsOptions>> {
    let Some(path) = options.obs_text_file.clone() else {
        if options.obs_max_lines.is_some()
            || options.obs_clear_on_start
            || options.obs_clear_on_stop
        {
            bail!(
                "--obs-max-lines/--obs-clear-on-start/--obs-clear-on-stop require --obs-text-file."
            );
        }
        return Ok(None);
    };
    validate_save_target_path(&path)?;
    let max_lines = options.obs_max_lines.unwrap_or(DEFAULT_OBS_MAX_LINES);
    if max_lines == 0 {
        bail!("--obs-max-lines must be greater than 0.");
    }
    Ok(Some(LiveObsOptions {
        path,
        max_lines,
        clear_on_start: options.obs_clear_on_start,
        clear_on_stop: options.obs_clear_on_stop,
    }))
}

fn build_markdown_note_options(
    options: &LiveCommandOptions<'_>,
    backend: BackendKind,
    model_id: &str,
    created_at: String,
) -> Result<Option<LiveMarkdownNoteOptions>> {
    let Some(path) = options.markdown_note_path.clone() else {
        if options.markdown_append
            || options.markdown_title.is_some()
            || options.markdown_suggest_title
        {
            bail!(
                "--markdown-append/--markdown-title/--markdown-suggest-title require --markdown-note."
            );
        }
        return Ok(None);
    };
    validate_save_target_path(&path)?;
    Ok(Some(LiveMarkdownNoteOptions {
        path,
        append: options.markdown_append,
        title: options.markdown_title.clone(),
        suggest_title: options.markdown_suggest_title,
        created_at,
        source_label: live_source_label(options.source).to_string(),
        model_id: model_id.to_string(),
        backend: backend_name(backend).to_string(),
    }))
}

fn live_source_label(source: LiveSource) -> &'static str {
    match source {
        LiveSource::Mic => "microphone",
        LiveSource::System => "system",
    }
}

fn validate_no_sink_path_collisions(
    save_options: Option<&LiveSaveOptions>,
    obs_options: Option<&LiveObsOptions>,
    markdown_note_options: Option<&LiveMarkdownNoteOptions>,
) -> Result<()> {
    let resolved_save = save_options
        .map(|save| resolve_sink_output_path(&save.path))
        .transpose()?;
    let resolved_obs = obs_options
        .map(|obs| resolve_sink_output_path(&obs.path))
        .transpose()?;
    let resolved_markdown = markdown_note_options
        .map(|markdown| resolve_sink_output_path(&markdown.path))
        .transpose()?;
    let resolved_markdown_partial = markdown_note_options
        .map(|markdown| resolve_sink_output_path(&implicit_partial_sidecar_path(&markdown.path)))
        .transpose()?;

    if let (Some(save), Some(_obs)) = (save_options, obs_options)
        && resolved_paths_collide(resolved_save.as_deref(), resolved_obs.as_deref())
    {
        bail!(
            "--save and --obs-text-file must not point to the same path: {}",
            save.path.display()
        );
    }
    if let (Some(save), Some(_markdown)) = (save_options, markdown_note_options)
        && resolved_paths_collide(resolved_save.as_deref(), resolved_markdown.as_deref())
    {
        bail!(
            "--save and --markdown-note must not point to the same path: {}",
            save.path.display()
        );
    }
    if let Some(save) = save_options {
        let partial_path = implicit_partial_sidecar_path(&save.path);
        let resolved_partial = resolve_sink_output_path(&partial_path)?;
        if obs_options.is_some()
            && resolved_paths_collide(Some(&resolved_partial), resolved_obs.as_deref())
        {
            bail!(
                "--obs-text-file must not point to --save's implicit partial sidecar path: {}",
                partial_path.display()
            );
        }
        if markdown_note_options.is_some() {
            if resolved_paths_collide(Some(&resolved_partial), resolved_markdown.as_deref()) {
                bail!(
                    "--markdown-note must not point to --save's implicit partial sidecar path: {}",
                    partial_path.display()
                );
            }
            if resolved_paths_collide(
                Some(&resolved_partial),
                resolved_markdown_partial.as_deref(),
            ) {
                bail!(
                    "--markdown-note partial sidecar must not point to --save's implicit partial sidecar path: {}",
                    partial_path.display()
                );
            }
        }
    }
    if let (Some(obs), Some(_markdown)) = (obs_options, markdown_note_options) {
        if resolved_paths_collide(resolved_obs.as_deref(), resolved_markdown.as_deref()) {
            bail!(
                "--obs-text-file and --markdown-note must not point to the same path: {}",
                obs.path.display()
            );
        }
        if resolved_paths_collide(
            resolved_obs.as_deref(),
            resolved_markdown_partial.as_deref(),
        ) {
            bail!(
                "--obs-text-file must not point to --markdown-note's implicit partial sidecar path: {}",
                obs.path.display()
            );
        }
    }
    if let (Some(save), Some(markdown)) = (save_options, markdown_note_options) {
        let markdown_partial = implicit_partial_sidecar_path(&markdown.path);
        if resolved_paths_collide(
            resolved_save.as_deref(),
            resolved_markdown_partial.as_deref(),
        ) {
            bail!(
                "--save must not point to --markdown-note's implicit partial sidecar path: {}",
                markdown_partial.display()
            );
        }
        let save_partial = resolve_sink_output_path(&implicit_partial_sidecar_path(&save.path))?;
        if resolved_paths_collide(Some(&save_partial), resolved_markdown_partial.as_deref()) {
            bail!(
                "--save implicit partial sidecar must not point to --markdown-note's implicit partial sidecar path: {}",
                markdown_partial.display()
            );
        }
    }
    Ok(())
}

fn implicit_partial_sidecar_path(save_path: &Path) -> PathBuf {
    let mut partial_path = save_path.as_os_str().to_os_string();
    partial_path.push(".partial");
    PathBuf::from(partial_path)
}

fn resolve_sink_output_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .context("Could not resolve current working directory for sink path validation")?
            .join(path)
    };
    if absolute.exists() {
        return absolute.canonicalize().with_context(|| {
            format!(
                "Could not canonicalize existing sink output path {}",
                absolute.display()
            )
        });
    }
    let parent = absolute.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "Could not resolve sink output parent directory for {}",
            absolute.display()
        )
    })?;
    let parent_canonical = parent.canonicalize().with_context(|| {
        format!(
            "Could not canonicalize sink output parent directory {}",
            parent.display()
        )
    })?;
    let file_name = absolute.file_name().ok_or_else(|| {
        anyhow::anyhow!(
            "Could not resolve sink output file name for {}",
            absolute.display()
        )
    })?;
    Ok(parent_canonical.join(file_name))
}

fn resolved_paths_collide(left: Option<&Path>, right: Option<&Path>) -> bool {
    let (Some(left), Some(right)) = (left, right) else {
        return false;
    };
    if left == right {
        return true;
    }
    let (Some(left_parent), Some(right_parent), Some(left_name), Some(right_name)) = (
        left.parent(),
        right.parent(),
        left.file_name(),
        right.file_name(),
    ) else {
        return false;
    };
    if left_parent != right_parent {
        return false;
    }
    let left_name = left_name.to_string_lossy();
    let right_name = right_name.to_string_lossy();
    if !left_name.eq_ignore_ascii_case(&right_name) {
        return false;
    }
    directory_is_case_insensitive(left_parent)
}

fn directory_is_case_insensitive(directory: &Path) -> bool {
    #[cfg(target_os = "windows")]
    {
        let _ = directory;
        true
    }

    #[cfg(not(target_os = "windows"))]
    {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        let probe_name = format!(".openasr_case_probe_{}_{}", std::process::id(), stamp);
        let probe_path = directory.join(&probe_name);
        let mut alternate_name = probe_name.to_uppercase();
        if alternate_name == probe_name {
            alternate_name = probe_name.to_lowercase();
        }
        let alternate_path = directory.join(alternate_name);
        let created = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&probe_path);
        if created.is_err() {
            return false;
        }
        let case_insensitive = alternate_path.exists();
        let _ = fs::remove_file(probe_path);
        case_insensitive
    }
}

fn validate_save_target_path(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
        if !parent.exists() {
            bail!(
                "Live save target parent directory does not exist: {}",
                parent.display()
            );
        }
        if !parent.is_dir() {
            bail!(
                "Live save target parent path is not a directory: {}",
                parent.display()
            );
        }
        if std::fs::metadata(parent)
            .map(|meta| meta.permissions().readonly())
            .unwrap_or(false)
        {
            bail!(
                "Live save target parent directory is read-only: {}",
                parent.display()
            );
        }
    }
    if path.exists() {
        let metadata = std::fs::metadata(path).with_context(|| {
            format!(
                "Could not inspect existing live save target at {}",
                path.display()
            )
        })?;
        if !metadata.is_file() {
            bail!(
                "Live save target must be a file path, but '{}' is not a file.",
                path.display()
            );
        }
        if metadata.permissions().readonly() {
            bail!("Live save target is read-only: {}", path.display());
        }
    }
    Ok(())
}

fn validate_live_limits(max_seconds: Option<u64>, max_utterances: Option<usize>) -> Result<()> {
    if max_seconds == Some(0) {
        bail!("--max-seconds must be greater than 0.");
    }
    if max_utterances == Some(0) {
        bail!("--max-utterances must be greater than 0.");
    }
    Ok(())
}

fn list_input_devices(host: &Host) -> Result<()> {
    let mut devices = host
        .input_devices()
        .context("Could not enumerate CPAL input devices")?
        .peekable();
    if devices.peek().is_none() {
        println!("No CPAL input devices were reported by the default host.");
        return Ok(());
    }

    for device in devices {
        println!("{}", device_label(&device));
        match device.supported_input_configs() {
            Ok(configs) => {
                for config in configs {
                    println!(
                        "  channels={} sample_format={} sample_rate={}..{}",
                        config.channels(),
                        config.sample_format(),
                        config.min_sample_rate(),
                        config.max_sample_rate()
                    );
                }
            }
            Err(error) => println!("  <could not query configs: {error}>"),
        }
    }
    Ok(())
}

fn list_system_audio_source() {
    let support = openasr_system_audio::support_status();
    println!("{}", support.label);
    println!("  platform={}", support.platform);
    println!("  supported={}", support.supported);
    println!("  detail={}", support.detail);
}

fn select_input_device(host: &Host, requested: Option<&str>) -> Result<Device> {
    if let Some(requested) = requested {
        let devices = host
            .input_devices()
            .context("Could not enumerate CPAL input devices")?;
        let mut fuzzy_match = None;
        for device in devices {
            let label = device_label(&device);
            if label == requested {
                return Ok(device);
            }
            if fuzzy_match.is_none() && label.to_lowercase().contains(&requested.to_lowercase()) {
                fuzzy_match = Some(device);
            }
        }
        return fuzzy_match.with_context(|| {
            format!("Input device '{requested}' was not found. Run `openasr live --source mic --list-devices` to inspect available devices.")
        });
    }

    host.default_input_device().with_context(|| {
        "No default microphone input device is available. Check OS microphone permissions/audio settings or run `openasr live --source mic --list-devices`.".to_string()
    })
}

fn device_label(device: &Device) -> String {
    device
        .description()
        .map(|description| description.to_string())
        .unwrap_or_else(|error| format!("<unknown input device: {error}>"))
}

fn build_input_stream(
    device: &Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    sender: SyncSender<CaptureChunk>,
    overflowed: &Arc<AtomicBool>,
) -> Result<Stream> {
    match sample_format {
        SampleFormat::F32 => {
            build_typed_input_stream::<f32>(device, config, sender, overflowed, CaptureChunk::F32)
        }
        SampleFormat::I16 => {
            build_typed_input_stream::<i16>(device, config, sender, overflowed, CaptureChunk::I16)
        }
        SampleFormat::U16 => {
            build_typed_input_stream::<u16>(device, config, sender, overflowed, CaptureChunk::U16)
        }
        other => bail!(
            "Unsupported microphone sample format '{other}'. M48B currently supports f32, i16, and u16 input samples."
        ),
    }
}

fn build_typed_input_stream<T>(
    device: &Device,
    config: &StreamConfig,
    sender: SyncSender<CaptureChunk>,
    overflowed: &Arc<AtomicBool>,
    wrap: fn(Vec<T>) -> CaptureChunk,
) -> Result<Stream>
where
    T: cpal::SizedSample + Clone + Send + 'static,
{
    let overflow_flag = Arc::clone(overflowed);
    let error_overflow_flag = Arc::clone(overflowed);
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                if data.is_empty() {
                    return;
                }
                match sender.try_send(wrap(data.to_vec())) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                        overflow_flag.store(true, Ordering::SeqCst);
                    }
                }
            },
            move |_error| {
                error_overflow_flag.store(true, Ordering::SeqCst);
            },
            None,
        )
        .context("Could not build CPAL microphone input stream")
}

#[derive(Debug)]
enum CaptureChunk {
    F32(Vec<f32>),
    I16(Vec<i16>),
    U16(Vec<u16>),
}

impl CaptureChunk {
    fn normalize_with(
        self,
        normalizer: &mut LiveAudioNormalizer,
    ) -> Result<Vec<RealtimeAudioFrame>> {
        match self {
            Self::F32(samples) => normalizer.push_f32_interleaved(&samples),
            Self::I16(samples) => normalizer.push_i16_interleaved(&samples),
            Self::U16(samples) => normalizer.push_u16_interleaved(&samples),
        }
    }
}

struct LiveCaptureRun {
    receiver: Receiver<CaptureChunk>,
    normalizer: LiveAudioNormalizer,
    overflowed: Arc<AtomicBool>,
    started_at: Instant,
    max_seconds: Option<u64>,
    max_utterances: Option<usize>,
    stop_requested: Arc<AtomicBool>,
    transcription_worker: LiveTranscriptionWorker,
}

impl LiveCaptureRun {
    fn run(&mut self, pipeline: &mut LivePipeline) -> Result<()> {
        loop {
            if let Err(error) =
                pipeline.drain_finished_transcriptions(&mut self.transcription_worker)
            {
                let _ = pipeline.shutdown(
                    self.normalizer.next_frame_start_ms(),
                    &mut self.transcription_worker,
                    false,
                );
                return Err(error);
            }
            if self.overflowed.swap(false, Ordering::SeqCst) {
                pipeline.emit_error(
                    RealtimeErrorCode::BackpressureTimeout,
                    "Microphone capture queue overflowed; OpenASR stopped instead of silently dropping audio.",
                    false,
                )?;
                let _ = pipeline.shutdown(
                    self.normalizer.next_frame_start_ms(),
                    &mut self.transcription_worker,
                    false,
                );
                bail!(
                    "Microphone capture queue overflowed; stopped instead of silently dropping audio."
                );
            }
            if self.stop_requested.load(Ordering::SeqCst) {
                break;
            }
            if self
                .max_seconds
                .is_some_and(|limit| self.started_at.elapsed() >= Duration::from_secs(limit))
            {
                break;
            }
            if self
                .max_utterances
                .is_some_and(|limit| pipeline.accepted_utterances >= limit)
            {
                break;
            }

            match self.receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => {
                    let frames = chunk.normalize_with(&mut self.normalizer)?;
                    for frame in frames {
                        if let Err(error) =
                            pipeline.process_frame(frame, &mut self.transcription_worker)
                        {
                            let _ = pipeline.shutdown(
                                self.normalizer.next_frame_start_ms(),
                                &mut self.transcription_worker,
                                false,
                            );
                            return Err(error);
                        }
                        if let Err(error) =
                            pipeline.drain_finished_transcriptions(&mut self.transcription_worker)
                        {
                            let _ = pipeline.shutdown(
                                self.normalizer.next_frame_start_ms(),
                                &mut self.transcription_worker,
                                false,
                            );
                            return Err(error);
                        }
                        if self
                            .max_utterances
                            .is_some_and(|limit| pipeline.accepted_utterances >= limit)
                        {
                            break;
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        pipeline.shutdown(
            self.normalizer.next_frame_start_ms(),
            &mut self.transcription_worker,
            true,
        )
    }
}

struct LiveTranscriptionJob {
    utterance_id: TranscriptUtteranceId,
    start_ms: u64,
    end_ms: u64,
    segment_id: TranscriptSegmentId,
    model_id: String,
    model_pack_path: Option<PathBuf>,
    display_name: String,
    temp_wav: tempfile::NamedTempFile,
    partial: bool,
    revision: u32,
    generation: u64,
}

struct LiveTranscriptionSuccess {
    utterance_id: TranscriptUtteranceId,
    start_ms: u64,
    end_ms: u64,
    segment_id: TranscriptSegmentId,
    text: String,
    partial: bool,
    revision: u32,
    generation: u64,
}

enum LiveTranscriptionResult {
    Final(LiveTranscriptionSuccess),
    Error(anyhow::Error),
}

struct LiveTranscriptionWorker {
    jobs: Option<SyncSender<LiveTranscriptionJob>>,
    results: Receiver<LiveTranscriptionResult>,
    handle: Option<JoinHandle<()>>,
    pending_jobs: usize,
}

impl LiveTranscriptionWorker {
    fn spawn(backend: BackendKind, model_pack_path: Option<PathBuf>) -> Self {
        let (job_sender, job_receiver) =
            mpsc::sync_channel::<LiveTranscriptionJob>(LIVE_TRANSCRIPTION_QUEUE_CAPACITY);
        let (result_sender, result_receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            for job in job_receiver {
                let result = transcribe_with_backend(
                    backend,
                    TranscriptionRequest::new(job.temp_wav.path(), job.model_id)
                        .with_source(openasr_core::RequestSource::CliLive)
                        // Not a normalization-pipeline guess: `write_temp_utterance_wav`
                        // above always writes PCM16 mono 16 kHz WAV -- this *is*
                        // the mic capture's real captured format/container,
                        // unlike an uploaded file whose source format is
                        // unknown until probed/decoded.
                        .with_source_audio_format(Some(16_000), Some(1))
                        .with_source_container(Some("wav".to_string()))
                        .with_model_pack_path(
                            job.model_pack_path.or_else(|| model_pack_path.clone()),
                        )
                        .with_display_file_name(Some(job.display_name)),
                )
                .map(|transcription| {
                    LiveTranscriptionResult::Final(LiveTranscriptionSuccess {
                        utterance_id: job.utterance_id,
                        start_ms: job.start_ms,
                        end_ms: job.end_ms,
                        segment_id: job.segment_id,
                        text: transcription.text,
                        partial: job.partial,
                        revision: job.revision,
                        generation: job.generation,
                    })
                })
                .unwrap_or_else(|error| {
                    LiveTranscriptionResult::Error(error.context(format!(
                        "Could not transcribe completed live utterance with the {} backend",
                        backend_name(backend)
                    )))
                });
                if result_sender.send(result).is_err() {
                    break;
                }
            }
        });

        Self {
            jobs: Some(job_sender),
            results: result_receiver,
            handle: Some(handle),
            pending_jobs: 0,
        }
    }

    fn queue(&mut self, job: LiveTranscriptionJob) -> Result<()> {
        let sender = self
            .jobs
            .as_ref()
            .context("Live transcription worker is not running")?;
        match sender.try_send(job) {
            Ok(()) => {
                self.pending_jobs += 1;
                Ok(())
            }
            Err(TrySendError::Full(_)) => {
                bail!(
                    "Live transcription worker queue is full; stopped instead of buffering unbounded utterances."
                )
            }
            Err(TrySendError::Disconnected(_)) => {
                bail!("Live transcription worker stopped before accepting the utterance.")
            }
        }
    }

    fn drain_available(&mut self) -> Vec<LiveTranscriptionResult> {
        let mut results = Vec::new();
        while let Ok(result) = self.results.try_recv() {
            self.pending_jobs = self.pending_jobs.saturating_sub(1);
            results.push(result);
        }
        results
    }

    fn finish(&mut self) -> Result<Vec<LiveTranscriptionResult>> {
        self.jobs.take();
        let mut results = Vec::new();
        while self.pending_jobs > 0 {
            match self.results.recv() {
                Ok(result) => {
                    self.pending_jobs -= 1;
                    results.push(result);
                }
                Err(_) => bail!("Live transcription worker stopped before returning all results."),
            }
        }
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("Live transcription worker panicked"))?;
        }
        Ok(results)
    }
}

impl Drop for LiveTranscriptionWorker {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[derive(Debug, Clone)]
struct LivePipelineConfig {
    model_id: String,
    model_pack_path: Option<PathBuf>,
    vad: VadConfig,
    buffer: RealtimeBufferConfig,
    partial_interval_ms: u64,
    partial_window_ms: u32,
    diarize: bool,
}

/// Resolve the live VAD mode. Delegates to the shared `openasr-core` resolver so
/// the CLI and the server WS never diverge: `OPENASR_VAD` wins, else **default to
/// the neural detector** (`OPENASR_VAD=energy`/`rms` opts out).
fn resolve_cli_vad_mode() -> VadMode {
    if openasr_core::diarize::vad::realtime_vad_prefers_neural(None) {
        VadMode::ExternalProbability
    } else {
        VadMode::Energy
    }
}

impl LivePipelineConfig {
    fn from_options(
        options: &LiveCommandOptions<'_>,
        model_id: String,
        model_pack_path: Option<PathBuf>,
    ) -> Result<Self> {
        openasr_core::RealtimeAudioFormat::pcm16_mono_16khz()
            .sample_count_for_duration_ms(options.frame_duration_ms)?;
        let mode = resolve_cli_vad_mode();
        // Source the per-field defaults from VadConfig::default() so the CLI does
        // not drift from the canonical realtime defaults over time.
        let default = VadConfig::default();
        // Mode-conditional hangover, matching the server WS so the two surfaces
        // endpoint identically: a neural session uses the shorter neural default,
        // the energy gate keeps 600. A `--speech-stop-ms` flag wins in either mode.
        let default_speech_stop_ms = match mode {
            VadMode::ExternalProbability => openasr_core::diarize::vad::SHORT_NEURAL_SPEECH_STOP_MS,
            _ => default.speech_stop_ms,
        };
        let vad = VadConfig {
            frame_duration_ms: options.frame_duration_ms,
            speech_start_ms: options.speech_start_ms.unwrap_or(default.speech_start_ms),
            speech_stop_ms: options.speech_stop_ms.unwrap_or(default_speech_stop_ms),
            pre_roll_ms: options.pre_roll_ms.unwrap_or(default.pre_roll_ms),
            max_utterance_ms: options.max_utterance_ms.or(default.max_utterance_ms),
            no_speech_timeout_ms: options
                .no_speech_timeout_ms
                .or(default.no_speech_timeout_ms),
            mode,
            energy_threshold: options.energy_threshold.unwrap_or(match mode {
                VadMode::ExternalProbability => {
                    openasr_core::diarize::vad::DEFAULT_NEURAL_VAD_THRESHOLD
                }
                _ => default.energy_threshold,
            }),
        };
        vad.validate()?;
        let max_utterance_ms = vad.max_utterance_ms.unwrap_or(30_000);
        let max_buffered_frames =
            ((max_utterance_ms + vad.pre_roll_ms) / options.frame_duration_ms + 2) as usize;
        let buffer = RealtimeBufferConfig {
            frame_duration_ms: options.frame_duration_ms,
            pre_roll_ms: vad.pre_roll_ms,
            max_buffered_frames: max_buffered_frames.max(DEFAULT_LIVE_MAX_BUFFERED_FRAMES),
            max_buffered_samples: 16_000 * ((max_utterance_ms + vad.pre_roll_ms + 1_000) as usize)
                / 1_000,
        };
        let partial_interval_ms = options
            .partial_interval_ms
            .unwrap_or(DEFAULT_STREAMING_PARTIAL_INTERVAL_MS);
        if partial_interval_ms == 0 {
            bail!("--partial-interval-ms must be greater than 0.");
        }
        let partial_window_ms = options
            .partial_window_ms
            .unwrap_or(DEFAULT_STREAMING_PARTIAL_WINDOW_MS);
        if partial_window_ms == 0 {
            bail!("--partial-window-ms must be greater than 0.");
        }
        Ok(Self {
            model_id,
            model_pack_path,
            vad,
            buffer,
            partial_interval_ms,
            partial_window_ms,
            diarize: options.diarize,
        })
    }
}

struct LivePipeline {
    controller: RealtimeSessionController,
    output_format: LiveOutputFormat,
    max_utterances: Option<usize>,
    accepted_utterances: usize,
    completed_utterances: usize,
    started_at: Instant,
    partial_emitted_count: usize,
    partial_suppressed_count: usize,
    partial_rollback_suppressed_count: usize,
    final_emitted_count: usize,
    first_partial_latency_ms: Option<u64>,
    first_final_latency_ms: Option<u64>,
    model_pack_path: Option<PathBuf>,
    partial_interval_ms: u64,
    partial_window_ms: u32,
    partial_flights: HashMap<TranscriptUtteranceId, PartialFlight>,
    /// Per-session streaming diarizer, built at startup when `--diarize` is
    /// requested; labels each finalized utterance with a stable anonymous
    /// speaker.
    streaming_diarizer: Option<openasr_core::diarize::streaming::StreamingDiarizer>,
    /// Speaker label per utterance, computed at queue time (has the audio) and
    /// consumed when the final transcript comes back.
    pending_utterance_speakers:
        HashMap<TranscriptUtteranceId, openasr_core::diarize::enrollment::SpeakerDisplayAssignment>,
    audio_running: bool,
    history: RealtimeTranscriptHistory,
    save_options: Option<LiveSaveOptions>,
    obs_options: Option<LiveObsOptions>,
    obs_sink_worker: Option<LiveObsSinkWorker>,
    markdown_note_options: Option<LiveMarkdownNoteOptions>,
    #[cfg(test)]
    emitted_events: Vec<RealtimeEventEnvelope>,
}

#[derive(Debug, Default)]
struct PartialFlight {
    in_flight: bool,
    pending_latest: Option<BufferedUtterance>,
    finalized: bool,
    generation: u64,
    last_emit_ms: u64,
    next_revision: u32,
    last_text: Option<String>,
}

impl PartialFlight {
    fn next_partial_revision(&mut self) -> u32 {
        self.next_revision = self.next_revision.saturating_add(1).max(1);
        self.next_revision
    }

    fn next_final_revision(&mut self) -> u32 {
        self.next_revision = self.next_revision.saturating_add(1).max(1);
        self.next_revision
    }

    fn invalidate_partials_for_final(&mut self) {
        self.finalized = true;
        self.pending_latest = None;
        self.in_flight = false;
        self.generation = self.generation.saturating_add(1);
    }
}

#[derive(Debug, Clone)]
struct LiveSaveOptions {
    path: PathBuf,
    format: RealtimeExportFormat,
    post_processor: RealtimePostProcessor,
}

#[derive(Debug, Clone)]
struct LiveObsOptions {
    path: PathBuf,
    max_lines: usize,
    clear_on_start: bool,
    clear_on_stop: bool,
}

#[derive(Debug)]
struct LiveObsSinkWorker {
    sender: Sender<String>,
    handle: JoinHandle<()>,
}

#[derive(Debug, Clone)]
struct LiveMarkdownNoteOptions {
    path: PathBuf,
    append: bool,
    title: Option<String>,
    suggest_title: bool,
    created_at: String,
    source_label: String,
    model_id: String,
    backend: String,
}

impl LivePipeline {
    fn new(
        config: LivePipelineConfig,
        output_format: LiveOutputFormat,
        max_utterances: Option<usize>,
        save_options: Option<LiveSaveOptions>,
    ) -> Result<Self> {
        let mut session =
            RealtimeSessionConfig::new(LIVE_SESSION_ID, config.model_id, timestamp_now());
        session.partial_results = true;
        session.vad = config.vad;
        session.buffer = config.buffer;
        session.diarize = config.diarize;
        // Build the diarizer up front so a pack that resolves but fails to
        // load aborts the session instead of silently degrading to anonymous
        // transcripts.
        let streaming_diarizer = if config.diarize {
            Some(
                openasr_core::diarize::streaming::StreamingDiarizer::shared(16_000).context(
                    "Live diarization was requested but the active speaker-embedder pack could not be loaded.",
                )?,
            )
        } else {
            None
        };
        Ok(Self {
            controller: RealtimeSessionController::new(session)?,
            output_format,
            max_utterances,
            accepted_utterances: 0,
            completed_utterances: 0,
            started_at: Instant::now(),
            partial_emitted_count: 0,
            partial_suppressed_count: 0,
            partial_rollback_suppressed_count: 0,
            final_emitted_count: 0,
            first_partial_latency_ms: None,
            first_final_latency_ms: None,
            model_pack_path: config.model_pack_path,
            partial_interval_ms: config.partial_interval_ms,
            partial_window_ms: config.partial_window_ms,
            partial_flights: HashMap::new(),
            streaming_diarizer,
            pending_utterance_speakers: HashMap::new(),
            audio_running: false,
            history: RealtimeTranscriptHistory::new(),
            save_options,
            obs_options: None,
            obs_sink_worker: None,
            markdown_note_options: None,
            #[cfg(test)]
            emitted_events: Vec::new(),
        })
    }

    fn configure_prototype_sinks(
        &mut self,
        obs_options: Option<LiveObsOptions>,
        markdown_note_options: Option<LiveMarkdownNoteOptions>,
    ) {
        self.obs_options = obs_options;
        self.markdown_note_options = markdown_note_options;
    }

    fn start(&mut self) -> Result<()> {
        self.started_at = Instant::now();
        self.maybe_clear_obs_on_start()?;
        self.start_obs_sink_worker();
        let created = self.controller.session_created_event(timestamp_now());
        self.emit(created)?;
        let configured = self
            .controller
            .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())?;
        self.emit(configured)?;
        let started = self
            .controller
            .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())?;
        self.audio_running = true;
        self.emit(started)
    }

    fn process_frame(
        &mut self,
        frame: RealtimeAudioFrame,
        transcription_worker: &mut LiveTranscriptionWorker,
    ) -> Result<()> {
        let frame_end_ms = frame.end_ms();
        let boundaries = self.controller.process_vad_frame(&frame);
        self.emit_boundaries(&boundaries)?;
        let utterances = match self.controller.buffer.push_frame(frame, &boundaries) {
            Ok(utterances) => utterances,
            Err(error) => {
                self.emit_error(
                    RealtimeErrorCode::AudioBufferOverflow,
                    &error.to_string(),
                    false,
                )?;
                return Err(error.into());
            }
        };
        self.maybe_queue_partial_snapshot(frame_end_ms, transcription_worker)?;
        for utterance in utterances {
            self.queue_utterance(utterance, transcription_worker)?;
        }
        Ok(())
    }

    fn maybe_queue_partial_snapshot(
        &mut self,
        now_ms: u64,
        transcription_worker: &mut LiveTranscriptionWorker,
    ) -> Result<()> {
        let Some(snapshot) = self
            .controller
            .buffer
            .active_snapshot(Some(self.partial_window_ms))
        else {
            return Ok(());
        };
        let utterance_id = snapshot.utterance_id.clone();
        let should_dispatch = {
            let flight = self
                .partial_flights
                .entry(utterance_id.clone())
                .or_default();
            if flight.finalized {
                return Ok(());
            }
            if now_ms.saturating_sub(flight.last_emit_ms) < self.partial_interval_ms {
                return Ok(());
            }
            flight.pending_latest = Some(snapshot);
            !flight.in_flight
        };
        if should_dispatch {
            self.queue_next_partial(&utterance_id, now_ms, transcription_worker)?;
        }
        Ok(())
    }

    fn queue_next_partial(
        &mut self,
        utterance_id: &TranscriptUtteranceId,
        now_ms: u64,
        transcription_worker: &mut LiveTranscriptionWorker,
    ) -> Result<()> {
        let Some(snapshot) = self
            .partial_flights
            .get(utterance_id)
            .and_then(|flight| flight.pending_latest.clone())
        else {
            return Ok(());
        };
        let temp_wav = write_temp_utterance_wav(&snapshot)?;
        let display_name = format!("{}.partial.wav", snapshot.utterance_id.0);
        let segment_id = TranscriptSegmentId(format!("{}_seg_000001", snapshot.utterance_id.0));
        let Some(flight) = self.partial_flights.get_mut(utterance_id) else {
            return Ok(());
        };
        if flight.finalized || flight.in_flight {
            return Ok(());
        }
        flight.pending_latest = None;
        flight.in_flight = true;
        flight.last_emit_ms = now_ms;
        let revision = flight.next_partial_revision();
        let generation = flight.generation;

        let queue_result = transcription_worker.queue(LiveTranscriptionJob {
            utterance_id: snapshot.utterance_id.clone(),
            start_ms: snapshot.start_ms,
            end_ms: snapshot.end_ms,
            segment_id,
            model_id: self.controller.config().model_id.clone(),
            model_pack_path: self.model_pack_path.clone(),
            display_name,
            temp_wav,
            partial: true,
            revision,
            generation,
        });
        if queue_result.is_err()
            && let Some(flight) = self.partial_flights.get_mut(utterance_id)
        {
            flight.in_flight = false;
            flight.pending_latest = Some(snapshot);
        }
        queue_result
    }

    fn shutdown(
        &mut self,
        end_ms: u64,
        transcription_worker: &mut LiveTranscriptionWorker,
        allow_final_save: bool,
    ) -> Result<()> {
        let outcome = (|| -> Result<()> {
            if let Some(utterance) = self.controller.buffer.flush(end_ms) {
                self.queue_utterance(utterance, transcription_worker)?;
            }
            let results = match transcription_worker.finish() {
                Ok(results) => results,
                Err(error) => {
                    self.emit_error(RealtimeErrorCode::BackendCrashed, &error.to_string(), false)?;
                    let _ = self.emit_terminal_events();
                    let _ = self.persist_partial_history_if_requested();
                    let _ = self.persist_markdown_note_on_error_if_requested();
                    return Err(error);
                }
            };
            let mut result_error = None;
            for result in results {
                if let Err(error) = self.apply_transcription_result(result, None) {
                    result_error = Some(error);
                    break;
                }
            }
            self.emit_terminal_events()?;
            if let Some(error) = result_error {
                let _ = self.persist_partial_history_if_requested();
                let _ = self.persist_markdown_note_on_error_if_requested();
                return Err(error);
            }
            if !allow_final_save {
                let _ = self.persist_partial_history_if_requested();
                let _ = self.persist_markdown_note_on_error_if_requested();
                return Ok(());
            }
            if let Err(error) = self.persist_history_if_requested() {
                let _ = self.persist_partial_history_if_requested();
                if let Err(note_error) = self.persist_markdown_note_if_requested() {
                    eprintln!(
                        "OpenASR live warning: could not save Markdown session note after --save failure: {note_error:#}"
                    );
                }
                return Err(error);
            }
            self.persist_markdown_note_if_requested()
        })();

        self.stop_obs_sink_worker();
        if let Err(clear_error) = self.maybe_clear_obs_on_stop() {
            eprintln!("OpenASR live warning: could not clear OBS text file on stop: {clear_error}");
        }
        self.print_streaming_metrics_summary();
        outcome
    }

    fn print_streaming_metrics_summary(&self) {
        let partial_revision_rate = if self.partial_emitted_count == 0 {
            0.0
        } else {
            self.partial_emitted_count
                .saturating_sub(self.final_emitted_count) as f64
                / self.partial_emitted_count as f64
        };
        eprintln!(
            "OpenASR live metrics: firstPartialLatencyMs={}, firstFinalLatencyMs={}, partialCount={}, finalCount={}, partialRevisionRate={:.4}, suppressedDuplicatePartials={}, suppressedRollbackPartials={}",
            self.first_partial_latency_ms
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            self.first_final_latency_ms
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            self.partial_emitted_count,
            self.final_emitted_count,
            partial_revision_rate,
            self.partial_suppressed_count,
            self.partial_rollback_suppressed_count
        );
    }

    fn emit_terminal_events(&mut self) -> Result<()> {
        if self.audio_running
            && self.controller.state() == openasr_core::RealtimeSessionState::Running
        {
            let stopped = self.controller.lifecycle(
                RealtimeLifecycleAction::StopAudio {
                    reason: "client_stopped".to_string(),
                },
                timestamp_now(),
            )?;
            self.emit(stopped)?;
            self.audio_running = false;
        }
        if !matches!(
            self.controller.state(),
            openasr_core::RealtimeSessionState::Closed
                | openasr_core::RealtimeSessionState::Cancelled
        ) {
            let closed = self.controller.lifecycle(
                RealtimeLifecycleAction::Close {
                    reason: "client_closed".to_string(),
                },
                timestamp_now(),
            )?;
            self.emit(closed)?;
        }
        Ok(())
    }

    fn drain_finished_transcriptions(
        &mut self,
        transcription_worker: &mut LiveTranscriptionWorker,
    ) -> Result<()> {
        for result in transcription_worker.drain_available() {
            self.apply_transcription_result(result, Some(transcription_worker))?;
        }
        Ok(())
    }

    fn emit_boundaries(&mut self, boundaries: &[SpeechBoundaryEvent]) -> Result<()> {
        for boundary in boundaries {
            match boundary {
                SpeechBoundaryEvent::SpeechStarted {
                    utterance_id,
                    start_ms,
                } => {
                    let envelope = self.controller.vad_event(
                        RealtimeVadEvent::SpeechStarted(openasr_core::VadSpeechStartedEvent {
                            utterance_id: utterance_id.clone(),
                            start_ms: *start_ms,
                        }),
                        timestamp_now(),
                    )?;
                    self.emit(envelope)?;
                }
                SpeechBoundaryEvent::SpeechStopped {
                    utterance_id,
                    start_ms,
                    end_ms,
                }
                | SpeechBoundaryEvent::MaxUtterance {
                    utterance_id,
                    start_ms,
                    end_ms,
                } => {
                    let envelope = self.controller.vad_event(
                        RealtimeVadEvent::SpeechStopped(openasr_core::VadSpeechStoppedEvent {
                            utterance_id: utterance_id.clone(),
                            start_ms: *start_ms,
                            end_ms: *end_ms,
                        }),
                        timestamp_now(),
                    )?;
                    self.emit(envelope)?;
                }
                SpeechBoundaryEvent::NoSpeechTimeout { timeout_ms, .. } => {
                    self.emit_error(
                        RealtimeErrorCode::NoSpeechTimeout,
                        &format!("No speech detected within {timeout_ms} ms."),
                        true,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn queue_utterance(
        &mut self,
        utterance: BufferedUtterance,
        transcription_worker: &mut LiveTranscriptionWorker,
    ) -> Result<()> {
        if utterance.reason == RealtimeUtteranceEndReason::Cancel {
            return Ok(());
        }
        if self
            .max_utterances
            .is_some_and(|limit| self.accepted_utterances >= limit)
        {
            return Ok(());
        }
        let temp_wav = write_temp_utterance_wav(&utterance)?;
        let display_name = format!("{}.wav", utterance.utterance_id.0);
        let segment_id = TranscriptSegmentId(format!("{}_seg_000001", utterance.utterance_id.0));
        let utterance_id = utterance.utterance_id.clone();
        // Diarize the utterance here, where its audio is still owned; the label
        // is attributed when the final transcript returns.
        if let Some(diarizer) = self.streaming_diarizer.as_mut() {
            let samples = utterance_samples_f32(&utterance.frames);
            if let Some(assignment) = diarizer.assign(&samples, 16_000) {
                self.pending_utterance_speakers
                    .insert(utterance_id.clone(), assignment);
            }
        }
        let flight = self
            .partial_flights
            .entry(utterance_id.clone())
            .or_default();
        // Compute the final's revision/generation but DON'T finalize the flight yet:
        // if the enqueue below fails, finalizing here would leave the flight
        // "finalized but never queued" and permanently suppress its partials.
        let generation = flight.generation;
        let revision = flight.next_final_revision();
        transcription_worker.queue(LiveTranscriptionJob {
            utterance_id: utterance.utterance_id,
            start_ms: utterance.start_ms,
            end_ms: utterance.end_ms,
            segment_id,
            model_id: self.controller.config().model_id.clone(),
            model_pack_path: self.model_pack_path.clone(),
            display_name,
            temp_wav,
            partial: false,
            revision,
            generation,
        })?;
        // Enqueue succeeded — now suppress any further partials for this utterance.
        if let Some(flight) = self.partial_flights.get_mut(&utterance_id) {
            flight.invalidate_partials_for_final();
        }
        self.accepted_utterances += 1;
        Ok(())
    }

    fn apply_transcription_result(
        &mut self,
        result: LiveTranscriptionResult,
        mut transcription_worker: Option<&mut LiveTranscriptionWorker>,
    ) -> Result<()> {
        let result = match result {
            LiveTranscriptionResult::Final(result) => result,
            LiveTranscriptionResult::Error(error) => {
                self.emit_error(RealtimeErrorCode::BackendCrashed, &error.to_string(), false)?;
                return Err(error);
            }
        };
        if result.partial {
            let flight = self
                .partial_flights
                .entry(result.utterance_id.clone())
                .or_default();
            flight.in_flight = false;
            if flight.finalized || result.generation != flight.generation {
                return Ok(());
            }
        } else if let Some(flight) = self.partial_flights.get_mut(&result.utterance_id) {
            if flight.finalized {
                flight.pending_latest = None;
                flight.in_flight = false;
            } else {
                flight.invalidate_partials_for_final();
            }
        }

        let speaker_assignment = if result.partial {
            None
        } else {
            self.pending_utterance_speakers.remove(&result.utterance_id)
        };
        let (speaker, speaker_label, speaker_profile_id) = speaker_assignment
            .map(|assignment| {
                let speaker_label = assignment
                    .speaker_profile_id
                    .is_some()
                    .then_some(assignment.speaker_label);
                (
                    Some(assignment.speaker),
                    speaker_label,
                    assignment.speaker_profile_id,
                )
            })
            .unwrap_or((None, None, None));
        let update = TranscriptUpdate {
            utterance_id: result.utterance_id.clone(),
            segment_id: result.segment_id,
            revision: result.revision as u64,
            text: result.text,
            start_ms: result.start_ms,
            end_ms: result.end_ms,
            language: None,
            speaker,
            speaker_label,
            speaker_profile_id,
            words: Vec::new(),
            revises_event_id: None,
        };
        if result.partial
            && let Some(previous) = self
                .partial_flights
                .get(&result.utterance_id)
                .and_then(|flight| flight.last_text.as_ref())
        {
            if previous == &update.text {
                self.partial_suppressed_count += 1;
                self.queue_pending_partial_after_result(
                    &result.utterance_id,
                    result.end_ms,
                    transcription_worker.as_deref_mut(),
                )?;
                return Ok(());
            }
            if should_suppress_partial_rollback(previous, &update.text) {
                self.partial_rollback_suppressed_count += 1;
                self.queue_pending_partial_after_result(
                    &result.utterance_id,
                    result.end_ms,
                    transcription_worker.as_deref_mut(),
                )?;
                return Ok(());
            }
        }
        let lifecycle = if result.partial {
            if let Some(flight) = self.partial_flights.get_mut(&result.utterance_id) {
                flight.last_text = Some(update.text.clone());
            }
            self.partial_emitted_count += 1;
            if self.first_partial_latency_ms.is_none() {
                self.first_partial_latency_ms = Some(self.started_at.elapsed().as_millis() as u64);
            }
            self.controller.transcript.apply_partial(update)
        } else {
            self.final_emitted_count += 1;
            if self.first_final_latency_ms.is_none() {
                self.first_final_latency_ms = Some(self.started_at.elapsed().as_millis() as u64);
            }
            self.controller.transcript.apply_final(update, None)
        };
        if let TranscriptLifecycleResult::Event(event) = lifecycle {
            let final_segment = match &event {
                RealtimeTranscriptEvent::Final(final_event) => Some((
                    final_event.utterance_id.clone(),
                    final_event.segment_id.clone(),
                    final_event.revision,
                )),
                _ => None,
            };
            let envelope = self.controller.transcript_event(event, timestamp_now())?;
            if let Some((utterance_id, segment_id, revision)) = final_segment {
                self.controller.transcript.record_final_event_id(
                    &utterance_id,
                    &segment_id,
                    revision,
                    envelope.event_id.clone(),
                );
            }
            self.emit(envelope)?;
            if !result.partial {
                self.completed_utterances += 1;
            }
        }
        if result.partial {
            self.queue_pending_partial_after_result(
                &result.utterance_id,
                result.end_ms,
                transcription_worker,
            )?;
        }
        Ok(())
    }

    fn queue_pending_partial_after_result(
        &mut self,
        utterance_id: &TranscriptUtteranceId,
        now_ms: u64,
        transcription_worker: Option<&mut LiveTranscriptionWorker>,
    ) -> Result<()> {
        if self
            .partial_flights
            .get(utterance_id)
            .is_some_and(|flight| !flight.finalized && flight.pending_latest.is_some())
            && let Some(transcription_worker) = transcription_worker
        {
            self.queue_next_partial(utterance_id, now_ms, transcription_worker)?;
        }
        Ok(())
    }

    fn emit_error(
        &mut self,
        code: RealtimeErrorCode,
        message: &str,
        recoverable: bool,
    ) -> Result<()> {
        let envelope = self.controller.error_event(
            RealtimeErrorEvent {
                code,
                message: message.to_string(),
                recoverable,
            },
            timestamp_now(),
        )?;
        self.emit(envelope)
    }

    fn emit(&mut self, envelope: RealtimeEventEnvelope) -> Result<()> {
        let refresh_obs = matches!(
            &envelope.event,
            openasr_core::RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(_))
                | openasr_core::RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(_))
        );

        #[cfg(test)]
        self.emitted_events.push(envelope.clone());

        let _ = self.history.apply_envelope(&envelope);

        match self.output_format {
            LiveOutputFormat::Jsonl => {
                println!("{}", serde_json::to_string(&envelope)?);
            }
            LiveOutputFormat::Text => match &envelope.event {
                openasr_core::RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) => {
                    match event.speaker.as_deref() {
                        Some(speaker) => println!("{speaker}: {}", event.text),
                        None => println!("{}", event.text),
                    }
                }
                openasr_core::RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(
                    event,
                )) => match event.speaker.as_deref() {
                    Some(speaker) => println!("{speaker}: {}", event.text),
                    None => println!("{}", event.text),
                },
                openasr_core::RealtimeEvent::Error(event) => {
                    eprintln!("OpenASR live error ({:?}): {}", event.code, event.message);
                }
                _ => {}
            },
        }
        if refresh_obs && let Err(error) = self.persist_obs_text_if_requested() {
            eprintln!("Warning: {error:#}");
        }
        Ok(())
    }

    fn persist_history_if_requested(&self) -> Result<()> {
        let Some(save) = self.save_options.as_ref() else {
            return Ok(());
        };
        let rendered = self
            .history
            .export(save.format, &save.post_processor)
            .with_context(|| {
                format!(
                    "Could not render live transcript history for {}",
                    save.path.display()
                )
            })?;
        atomic_write_text(&save.path, &rendered).with_context(|| {
            format!(
                "Could not save live transcript history to {}",
                save.path.display()
            )
        })?;
        eprintln!(
            "Saved live transcript history to {} ({:?}).",
            save.path.display(),
            save.format
        );
        Ok(())
    }

    fn persist_partial_history_if_requested(&self) -> Result<()> {
        let Some(save) = self.save_options.as_ref() else {
            return Ok(());
        };
        if self.history.entries().is_empty() {
            return Ok(());
        }
        let fallback_format = match save.format {
            RealtimeExportFormat::Srt | RealtimeExportFormat::Vtt => RealtimeExportFormat::Text,
            other => other,
        };
        let partial_path = implicit_partial_sidecar_path(&save.path);
        validate_save_target_path(&partial_path)?;
        if partial_path.exists() {
            bail!(
                "Refusing to overwrite implicit partial live transcript sidecar at {}. Remove it or choose a different --save path.",
                partial_path.display()
            );
        }
        let rendered = self
            .history
            .export(fallback_format, &save.post_processor)
            .with_context(|| {
                format!(
                    "Could not render partial live transcript history for {}",
                    partial_path.display()
                )
            })?;
        atomic_write_text(&partial_path, &rendered).with_context(|| {
            format!(
                "Could not save partial live transcript history to {}",
                partial_path.display()
            )
        })?;
        eprintln!(
            "Saved partial live transcript history to {} ({:?}) because live session ended with an error.",
            partial_path.display(),
            fallback_format
        );
        Ok(())
    }

    fn maybe_clear_obs_on_start(&self) -> Result<()> {
        let Some(obs) = self.obs_options.as_ref() else {
            return Ok(());
        };
        if !obs.clear_on_start {
            return Ok(());
        }
        atomic_write_text(&obs.path, "").with_context(|| {
            format!(
                "Could not clear OBS text file on start at {}",
                obs.path.display()
            )
        })
    }

    fn maybe_clear_obs_on_stop(&self) -> Result<()> {
        let Some(obs) = self.obs_options.as_ref() else {
            return Ok(());
        };
        if !obs.clear_on_stop {
            return Ok(());
        }
        atomic_write_text(&obs.path, "").with_context(|| {
            format!(
                "Could not clear OBS text file on stop at {}",
                obs.path.display()
            )
        })
    }

    fn persist_obs_text_if_requested(&self) -> Result<()> {
        let Some(obs) = self.obs_options.as_ref() else {
            return Ok(());
        };
        let Some(worker) = self.obs_sink_worker.as_ref() else {
            return Ok(());
        };
        let processed = self.history.post_process(&RealtimePostProcessor::default());
        let line_count = processed.lines.len();
        let start = line_count.saturating_sub(obs.max_lines);
        let content = if start >= line_count {
            String::new()
        } else {
            format!("{}\n", processed.lines[start..].join("\n"))
        };
        worker.sender.send(content).with_context(|| {
            format!(
                "Could not enqueue OBS text file prototype update for {}",
                obs.path.display()
            )
        })
    }

    fn persist_markdown_note_if_requested(&self) -> Result<()> {
        let Some(note) = self.markdown_note_options.as_ref() else {
            return Ok(());
        };
        let rendered = self.render_markdown_note(note);
        if !note.append {
            return atomic_write_text(&note.path, &rendered).with_context(|| {
                format!(
                    "Could not save Markdown session note prototype to {}",
                    note.path.display()
                )
            });
        }
        let existing = match fs::read_to_string(&note.path) {
            Ok(content) => content,
            Err(error) if error.kind() == io::ErrorKind::NotFound => String::new(),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "Could not read existing Markdown note for append at {}",
                        note.path.display()
                    )
                });
            }
        };
        let merged = if existing.is_empty() {
            rendered
        } else if existing.ends_with('\n') {
            format!("{existing}\n{rendered}")
        } else {
            format!("{existing}\n\n{rendered}")
        };
        atomic_write_text(&note.path, &merged).with_context(|| {
            format!(
                "Could not append Markdown session note prototype to {}",
                note.path.display()
            )
        })
    }

    fn persist_markdown_note_on_error_if_requested(&self) -> Result<()> {
        let Some(note) = self.markdown_note_options.as_ref() else {
            return Ok(());
        };
        let partial_path = implicit_partial_sidecar_path(&note.path);
        validate_save_target_path(&partial_path)?;
        if partial_path.exists() {
            bail!(
                "Refusing to overwrite implicit partial Markdown note sidecar at {}. Remove it or choose a different --markdown-note path.",
                partial_path.display()
            );
        }
        let rendered = self.render_markdown_note(note);
        atomic_write_text(&partial_path, &rendered).with_context(|| {
            format!(
                "Could not save partial Markdown session note prototype to {}",
                partial_path.display()
            )
        })?;
        eprintln!(
            "Saved partial Markdown session note prototype to {} because live session ended with an error.",
            partial_path.display()
        );
        Ok(())
    }

    fn start_obs_sink_worker(&mut self) {
        let Some(obs) = self.obs_options.as_ref() else {
            return;
        };
        if self.obs_sink_worker.is_some() {
            return;
        }
        let path = obs.path.clone();
        let (sender, receiver) = mpsc::channel::<String>();
        let handle = thread::spawn(move || {
            while let Ok(content) = receiver.recv() {
                if let Err(error) = atomic_write_text(&path, &content) {
                    eprintln!(
                        "OpenASR live warning: could not update OBS text file prototype at {}: {error:#}",
                        path.display()
                    );
                }
            }
        });
        self.obs_sink_worker = Some(LiveObsSinkWorker { sender, handle });
    }

    fn stop_obs_sink_worker(&mut self) {
        let Some(worker) = self.obs_sink_worker.take() else {
            return;
        };
        drop(worker.sender);
        if worker.handle.join().is_err() {
            eprintln!("OpenASR live warning: OBS text file worker terminated unexpectedly.");
        }
    }

    fn render_markdown_note(&self, note: &LiveMarkdownNoteOptions) -> String {
        let post = RealtimePostProcessor {
            suggest_title: note.suggest_title,
            ..RealtimePostProcessor::default()
        };
        let processed = self.history.post_process(&post);
        let title = note
            .title
            .clone()
            .or(processed.title.clone())
            .unwrap_or_else(|| "OpenASR Live Session".to_string());
        let title = escape_markdown_text(&normalize_markdown_line(&title));
        let mut lines = Vec::new();
        lines.push(format!("# {title}"));
        lines.push(String::new());
        lines.push(format!("- created_at: {}", note.created_at));
        lines.push(format!("- source: {}", note.source_label));
        lines.push(format!("- model: {}", note.model_id));
        lines.push(format!("- backend: {}", note.backend));
        lines.push(format!(
            "- finalized_lines: {}",
            self.history.entries().len()
        ));
        lines.push(format!("- revisions: {}", self.history.revisions().len()));
        lines.push(String::new());
        lines.push("## Transcript".to_string());
        lines.push(String::new());
        if processed.lines.is_empty() {
            lines.push("_No finalized transcript lines._".to_string());
        } else {
            lines.extend(
                processed
                    .lines
                    .iter()
                    .map(|line| format!("- {}", escape_markdown_text(line))),
            );
        }
        lines.push(String::new());
        lines.push("## Timeline".to_string());
        lines.push(String::new());
        let mut sorted = self.history.entries().iter().collect::<Vec<_>>();
        sorted.sort_by(|left, right| {
            left.start_ms
                .cmp(&right.start_ms)
                .then_with(|| left.end_ms.cmp(&right.end_ms))
                .then_with(|| {
                    left.source_seq
                        .unwrap_or(u64::MAX)
                        .cmp(&right.source_seq.unwrap_or(u64::MAX))
                })
                .then_with(|| left.utterance_id.cmp(&right.utterance_id))
                .then_with(|| left.segment_id.cmp(&right.segment_id))
        });
        if sorted.is_empty() {
            lines.push("_No finalized/revised timeline entries._".to_string());
        } else {
            lines.extend(sorted.into_iter().map(|entry| {
                let normalized_text = escape_markdown_text(&normalize_markdown_line(&entry.text));
                format!(
                    "- {}-{} ms | {} | rev {} | {}",
                    entry.start_ms, entry.end_ms, entry.segment_id, entry.revision, normalized_text
                )
            }));
        }
        format!("{}\n", lines.join("\n"))
    }
}

fn should_suppress_partial_rollback(previous: &str, next: &str) -> bool {
    let prev = previous.trim();
    let next = next.trim();
    if prev.is_empty() || next.is_empty() {
        return false;
    }
    if next.len() >= prev.len() {
        return false;
    }
    if !prev.starts_with(next) {
        return false;
    }
    prev.len().saturating_sub(next.len()) >= PARTIAL_ROLLBACK_SUPPRESS_MIN_DELTA_CHARS
}

fn normalize_markdown_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn escape_markdown_text(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        if matches!(
            ch,
            '\\' | '`'
                | '*'
                | '_'
                | '{'
                | '}'
                | '['
                | ']'
                | '('
                | ')'
                | '#'
                | '+'
                | '-'
                | '!'
                | '|'
                | '>'
        ) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

/// Thin cpal-facing wrapper around the shared, platform-agnostic
/// `openasr_core::realtime::CaptureEngine` (resample/downmix/frame). The CLI
/// owns the OS audio API (cpal device selection, stream lifecycle); the
/// actual PCM normalization logic lives once in `openasr-core` so the
/// desktop live path and the (open-core, cross-repo) mobile capture engine
/// never drift.
#[derive(Debug)]
struct LiveAudioNormalizer {
    engine: openasr_core::CaptureEngine,
}

impl LiveAudioNormalizer {
    fn new(input_sample_rate_hz: u32, input_channels: u16, frame_duration_ms: u32) -> Result<Self> {
        let input = openasr_core::CaptureInputFormat::new(input_sample_rate_hz, input_channels)?;
        let engine = openasr_core::CaptureEngine::new(input, frame_duration_ms)?;
        Ok(Self { engine })
    }

    fn push_f32_interleaved(&mut self, samples: &[f32]) -> Result<Vec<RealtimeAudioFrame>> {
        Ok(self.engine.push_f32_interleaved(samples)?)
    }

    fn push_i16_interleaved(&mut self, samples: &[i16]) -> Result<Vec<RealtimeAudioFrame>> {
        Ok(self.engine.push_i16_interleaved(samples)?)
    }

    fn push_u16_interleaved(&mut self, samples: &[u16]) -> Result<Vec<RealtimeAudioFrame>> {
        Ok(self.engine.push_u16_interleaved(samples)?)
    }

    fn next_frame_start_ms(&self) -> u64 {
        self.engine.next_frame_start_ms()
    }
}

/// Flatten an utterance's PCM16 frames to `[-1, 1]` `f32` mono for embedding.
fn utterance_samples_f32(frames: &[RealtimeAudioFrame]) -> Vec<f32> {
    frames
        .iter()
        .flat_map(|frame| frame.samples().iter().map(|s| *s as f32 / 32768.0))
        .collect()
}

fn write_temp_utterance_wav(utterance: &BufferedUtterance) -> Result<tempfile::NamedTempFile> {
    let mut file = tempfile::Builder::new()
        .prefix(LIVE_TEMP_PREFIX)
        .suffix(".wav")
        .tempfile()
        .context("Could not create temporary live utterance WAV file")?;
    let samples = utterance
        .frames
        .iter()
        .flat_map(|frame| frame.samples().iter().copied())
        .collect::<Vec<_>>();
    write_pcm16_mono_16khz_wav(file.as_file_mut(), &samples)?;
    file.as_file_mut()
        .flush()
        .context("Could not flush temporary live utterance WAV file")?;
    Ok(file)
}

fn write_pcm16_mono_16khz_wav(mut writer: impl Write, samples: &[i16]) -> io::Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let riff_len = 36u32 + data_len;
    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_len.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&16_000u32.to_le_bytes())?;
    writer.write_all(&32_000u32.to_le_bytes())?;
    writer.write_all(&2u16.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
    writer.write_all(b"data")?;
    writer.write_all(&data_len.to_le_bytes())?;
    for sample in samples {
        writer.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}

fn timestamp_now() -> String {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => format_unix_millis(duration.as_secs(), duration.subsec_millis()),
        Err(_) => "1970-01-01T00:00:00.000Z".to_string(),
    }
}

fn format_unix_millis(seconds: u64, millis: u32) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    (year, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use openasr_core::{
        RealtimeAudioFormat, RealtimeAudioFrame, RealtimeTranscriptFinal,
        RealtimeTranscriptPartial, RealtimeTranscriptRevision, TranscriptSegmentId,
        TranscriptUtteranceId,
    };

    fn test_live_config() -> LivePipelineConfig {
        LivePipelineConfig {
            model_id: "whisper-large-v3-turbo".to_string(),
            model_pack_path: None,
            diarize: false,
            vad: VadConfig {
                frame_duration_ms: 20,
                speech_start_ms: 40,
                speech_stop_ms: 40,
                pre_roll_ms: 20,
                max_utterance_ms: Some(1_000),
                no_speech_timeout_ms: None,
                mode: VadMode::Energy,
                energy_threshold: 0.02,
            },
            buffer: RealtimeBufferConfig {
                frame_duration_ms: 20,
                pre_roll_ms: 20,
                max_buffered_frames: 20,
                max_buffered_samples: 10_000,
            },
            partial_interval_ms: DEFAULT_STREAMING_PARTIAL_INTERVAL_MS,
            partial_window_ms: DEFAULT_STREAMING_PARTIAL_WINDOW_MS,
        }
    }

    fn mock_transcription_worker() -> LiveTranscriptionWorker {
        LiveTranscriptionWorker::spawn(BackendKind::Mock, None)
    }

    fn frame(seq: u64, start_ms: u64, sample: i16) -> RealtimeAudioFrame {
        RealtimeAudioFrame::new(
            seq,
            start_ms,
            RealtimeAudioFormat::pcm16_mono_16khz(),
            vec![sample; 320],
        )
        .unwrap()
    }

    fn buffered_utterance(
        utterance_id: &str,
        seq: u64,
        start_ms: u64,
        sample: i16,
    ) -> BufferedUtterance {
        BufferedUtterance {
            utterance_id: TranscriptUtteranceId(utterance_id.to_string()),
            start_ms,
            end_ms: start_ms + 20,
            frames: vec![frame(seq, start_ms, sample)],
            reason: RealtimeUtteranceEndReason::Flush,
        }
    }

    #[test]
    fn live_backend_ready_allows_mock() {
        ensure_live_backend_ready(BackendKind::Mock, None).unwrap();
    }

    #[test]
    fn live_backend_ready_rejects_native_placeholder() {
        let error = ensure_live_backend_ready(BackendKind::Native, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires --model-pack"));
        assert!(error.contains("fail-closed"));
    }

    #[test]
    fn live_model_pack_resolution_matches_transcribe_for_catalog_aliases() {
        // Regression for the `openasr live` catalog-blindness bug: resolving an
        // installed pack by a catalog family:tag alias (e.g. `qwen:q8`) must
        // succeed the same way it does for `transcribe`'s
        // `resolve_model_source_for_backend`, which forwards the loaded
        // catalog into `resolve_installed_native_pack`. Before the fix, `live`
        // hardcoded `None` for the catalog there, so the same alias that
        // installed cleanly for `transcribe` was reported "not installed" for
        // `live`.
        let temp = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("OPENASR_HOME", temp.path()) };
        let home = openasr_home().unwrap();
        let config = openasr_core::OpenAsrConfig::default();
        let catalog = load_cli_model_catalog(&home).unwrap();
        assert!(
            catalog.is_some(),
            "repo-local model-registry/catalog.json must load"
        );

        // Hand-write an installed-pack record directly (bypassing the real
        // download/signature-verified install path, which is orthogonal to
        // this alias-resolution regression): `list_installed_packs` needs
        // `models/<model_id>/<quant>/installed.json` plus a same-named pack
        // file on disk that passes `validate_native_runtime_model_pack_contract`,
        // so the pack payload itself must be a structurally valid (if tiny)
        // runtime pack -- the registry-facing `model_id` label is independent
        // of the pack's internal `openasr.*` GGUF metadata, so a generic
        // one-layer fixture works under the "qwen3-asr-0.6b" registry id.
        let model_id = "qwen3-asr-0.6b";
        let quant = "q8_0";
        let pack_dir = home.join("models").join(model_id).join(quant);
        fs::create_dir_all(&pack_dir).unwrap();
        let pack_filename = format!("{model_id}-{quant}.oasr");
        let pack_path = pack_dir.join(&pack_filename);
        let fixture_spec =
            openasr_core::testing::TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer(
                model_id,
            );
        openasr_core::testing::write_tiny_gguf_runtime_source(&pack_path, &fixture_spec).unwrap();
        let installed = openasr_core::InstalledPack {
            model_id: model_id.to_string(),
            display_name: "Qwen3-ASR 0.6B".to_string(),
            quant: quant.to_string(),
            suffix: "q8".to_string(),
            pull: format!("{model_id}:{quant}"),
            filename: pack_filename,
            path: pack_path.clone(),
            url: "https://example.invalid/qwen3-asr-0.6b-q8_0.oasr".to_string(),
            hf_revision: "0".repeat(40),
            sha256: "0".repeat(64),
            size_bytes: fs::metadata(&pack_path).unwrap().len(),
            installed_at_unix_seconds: 0,
            source: None,
        };
        fs::write(
            pack_dir.join("installed.json"),
            serde_json::to_string(&installed).unwrap(),
        )
        .unwrap();

        // Sanity check: the `qwen:q8` shorthand only resolves through the
        // catalog (family alias "qwen" -> canonical id "qwen3-asr-0.6b");
        // catalog-blind resolution must fail to prove the alias genuinely
        // needs the catalog forwarded.
        let without_catalog = crate::native_segment_cli::resolve_installed_native_pack(
            Some("qwen:q8"),
            &config,
            None,
        );
        assert!(
            without_catalog.is_err(),
            "catalog-blind resolution unexpectedly matched the installed pack"
        );

        let transcribe_source = crate::native_segment_cli::resolve_model_source_for_backend(
            "transcription",
            Some("qwen:q8"),
            BackendKind::Native,
            None,
            &config,
        )
        .expect("transcribe resolution must resolve the qwen:q8 alias via the catalog");
        assert_eq!(
            transcribe_source.model_pack_path,
            Some(installed.path.clone())
        );

        let live_pack = resolve_live_model_pack(
            BackendKind::Native,
            Some("qwen:q8"),
            &config,
            None,
            catalog.as_ref(),
        )
        .expect(
            "live resolution must resolve the qwen:q8 alias via the catalog, matching transcribe",
        );
        assert_eq!(live_pack, Some(installed.path));
    }

    fn assert_eventually_file_equals(path: &Path, expected: &str) {
        let mut last = String::new();
        for _ in 0..100 {
            if let Ok(content) = std::fs::read_to_string(path) {
                if content == expected {
                    return;
                }
                last = content;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(last, expected);
    }

    #[test]
    fn parses_live_source_and_format() {
        assert_eq!(parse_live_source("mic"), Ok(LiveSource::Mic));
        assert_eq!(parse_live_source("system"), Ok(LiveSource::System));
        assert!(
            parse_live_source("loopback")
                .unwrap_err()
                .contains("mic, system")
        );
        assert_eq!(parse_live_output_format("text"), Ok(LiveOutputFormat::Text));
        assert_eq!(
            parse_live_output_format("jsonl"),
            Ok(LiveOutputFormat::Jsonl)
        );
        assert!(
            parse_live_output_format("ndjson")
                .unwrap_err()
                .contains("text, jsonl")
        );
    }

    #[test]
    fn validates_live_limits() {
        validate_live_limits(Some(1), Some(1)).unwrap();
        assert!(
            validate_live_limits(Some(0), Some(1))
                .unwrap_err()
                .to_string()
                .contains("--max-seconds")
        );
        assert!(
            validate_live_limits(Some(1), Some(0))
                .unwrap_err()
                .to_string()
                .contains("--max-utterances")
        );
    }

    fn base_live_options<'a>() -> LiveCommandOptions<'a> {
        LiveCommandOptions {
            source: LiveSource::Mic,
            list_devices: false,
            device: None,
            input_file: None,
            model: None,
            backend: Some(BackendKind::Mock),
            model_pack: None,
            diarize: false,
            output_format: LiveOutputFormat::Text,
            max_seconds: None,
            max_utterances: None,
            frame_duration_ms: 20,
            speech_start_ms: None,
            speech_stop_ms: None,
            pre_roll_ms: None,
            max_utterance_ms: None,
            no_speech_timeout_ms: None,
            energy_threshold: None,
            partial_interval_ms: None,
            partial_window_ms: None,
            save_path: None,
            save_join_segments: false,
            save_suggest_title: false,
            obs_text_file: None,
            obs_max_lines: None,
            obs_clear_on_start: false,
            obs_clear_on_stop: false,
            markdown_note_path: None,
            markdown_append: false,
            markdown_title: None,
            markdown_suggest_title: false,
            runtime_paths: RuntimePathOverrides::default(),
            consent: crate::consent::PullConsent::default(),
        }
    }

    #[test]
    fn validates_live_save_extension() {
        let temp = tempfile::tempdir().unwrap();
        let mut options = base_live_options();
        options.save_path = Some(temp.path().join("live.txt"));
        assert!(build_save_options(&options).unwrap().is_some());

        options.save_path = Some(temp.path().join("live.bin"));
        let error = build_save_options(&options).unwrap_err().to_string();
        assert!(error.contains("Unsupported live export extension"));
    }

    #[test]
    fn validates_live_save_target_path_preconditions() {
        let temp = tempfile::tempdir().unwrap();
        let mut options = base_live_options();
        options.save_path = Some(temp.path().join("missing").join("live.txt"));
        let missing_parent = build_save_options(&options).unwrap_err().to_string();
        assert!(missing_parent.contains("parent directory does not exist"));

        let target_dir = temp.path().join("dir.txt");
        std::fs::create_dir_all(&target_dir).unwrap();
        options.save_path = Some(target_dir);
        let non_file = build_save_options(&options).unwrap_err().to_string();
        assert!(non_file.contains("must be a file path"));
    }

    #[test]
    fn validates_obs_and_markdown_option_preconditions() {
        let temp = tempfile::tempdir().unwrap();
        let mut options = base_live_options();
        options.obs_max_lines = Some(2);
        let obs_missing_path = build_obs_options(&options).unwrap_err().to_string();
        assert!(obs_missing_path.contains("require --obs-text-file"));

        options = base_live_options();
        options.obs_text_file = Some(temp.path().join("missing").join("obs.txt"));
        let obs_error = build_obs_options(&options).unwrap_err().to_string();
        assert!(obs_error.contains("parent directory does not exist"));

        options.obs_text_file = Some(temp.path().join("obs.txt"));
        options.obs_max_lines = Some(0);
        let obs_lines_error = build_obs_options(&options).unwrap_err().to_string();
        assert!(obs_lines_error.contains("--obs-max-lines"));

        options.obs_max_lines = Some(2);
        options.markdown_append = true;
        let markdown_missing_path = build_markdown_note_options(
            &options,
            BackendKind::Mock,
            "whisper-large-v3-turbo",
            "2026-05-10T00:00:00.000Z".to_string(),
        )
        .unwrap_err()
        .to_string();
        assert!(markdown_missing_path.contains("require --markdown-note"));

        options.markdown_append = false;
        options.markdown_note_path = Some(temp.path().join("note.md"));
        assert!(
            build_markdown_note_options(
                &options,
                BackendKind::Mock,
                "whisper-large-v3-turbo",
                "2026-05-10T00:00:00.000Z".to_string()
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn rejects_overlapping_sink_output_paths() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("shared.txt");
        let save = LiveSaveOptions {
            path: path.clone(),
            format: RealtimeExportFormat::Text,
            post_processor: RealtimePostProcessor::default(),
        };
        let obs = LiveObsOptions {
            path: path.clone(),
            max_lines: 2,
            clear_on_start: false,
            clear_on_stop: false,
        };
        let markdown = LiveMarkdownNoteOptions {
            path,
            append: false,
            title: None,
            suggest_title: false,
            created_at: "2026-05-10T00:00:00.000Z".to_string(),
            source_label: "microphone".to_string(),
            model_id: "whisper-large-v3-turbo".to_string(),
            backend: "mock".to_string(),
        };
        let error = validate_no_sink_path_collisions(Some(&save), Some(&obs), Some(&markdown))
            .unwrap_err()
            .to_string();
        assert!(error.contains("--save and --obs-text-file"));
    }

    #[test]
    fn rejects_sink_paths_colliding_with_partial_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let save_path = temp.path().join("session.txt");
        let save = LiveSaveOptions {
            path: save_path.clone(),
            format: RealtimeExportFormat::Text,
            post_processor: RealtimePostProcessor::default(),
        };

        let obs = LiveObsOptions {
            path: PathBuf::from(format!("{}.partial", save_path.display())),
            max_lines: 2,
            clear_on_start: false,
            clear_on_stop: false,
        };
        let obs_error = validate_no_sink_path_collisions(Some(&save), Some(&obs), None)
            .unwrap_err()
            .to_string();
        assert!(obs_error.contains("implicit partial sidecar path"));

        let markdown = LiveMarkdownNoteOptions {
            path: PathBuf::from(format!("{}.partial", save_path.display())),
            append: false,
            title: None,
            suggest_title: false,
            created_at: "2026-05-10T00:00:00.000Z".to_string(),
            source_label: "microphone".to_string(),
            model_id: "whisper-large-v3-turbo".to_string(),
            backend: "mock".to_string(),
        };
        let markdown_error = validate_no_sink_path_collisions(Some(&save), None, Some(&markdown))
            .unwrap_err()
            .to_string();
        assert!(markdown_error.contains("implicit partial sidecar path"));
    }

    #[test]
    fn rejects_path_alias_collisions() {
        let temp = tempfile::tempdir().unwrap();
        let save_path = temp.path().join("session.txt");
        let save = LiveSaveOptions {
            path: save_path.clone(),
            format: RealtimeExportFormat::Text,
            post_processor: RealtimePostProcessor::default(),
        };
        let obs = LiveObsOptions {
            path: temp.path().join(".").join("session.txt"),
            max_lines: 2,
            clear_on_start: false,
            clear_on_stop: false,
        };
        let error = validate_no_sink_path_collisions(Some(&save), Some(&obs), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--save and --obs-text-file"));
    }

    #[test]
    fn rejects_markdown_partial_sidecar_collisions() {
        let temp = tempfile::tempdir().unwrap();
        let note_path = temp.path().join("session.md");
        let markdown = LiveMarkdownNoteOptions {
            path: note_path.clone(),
            append: false,
            title: None,
            suggest_title: false,
            created_at: "2026-05-10T00:00:00.000Z".to_string(),
            source_label: "microphone".to_string(),
            model_id: "whisper-large-v3-turbo".to_string(),
            backend: "mock".to_string(),
        };

        let save = LiveSaveOptions {
            path: PathBuf::from(format!("{}.partial", note_path.display())),
            format: RealtimeExportFormat::Text,
            post_processor: RealtimePostProcessor::default(),
        };
        let save_error = validate_no_sink_path_collisions(Some(&save), None, Some(&markdown))
            .unwrap_err()
            .to_string();
        assert!(save_error.contains("--markdown-note's implicit partial sidecar path"));

        let obs = LiveObsOptions {
            path: PathBuf::from(format!("{}.partial", note_path.display())),
            max_lines: 2,
            clear_on_start: false,
            clear_on_stop: false,
        };
        let obs_error = validate_no_sink_path_collisions(None, Some(&obs), Some(&markdown))
            .unwrap_err()
            .to_string();
        assert!(obs_error.contains("--markdown-note's implicit partial sidecar path"));
    }

    #[test]
    fn handles_case_only_collisions_based_on_filesystem_behavior() {
        let temp = tempfile::tempdir().unwrap();
        let save_path = temp.path().join("session.txt");
        let save = LiveSaveOptions {
            path: save_path,
            format: RealtimeExportFormat::Text,
            post_processor: RealtimePostProcessor::default(),
        };
        let obs = LiveObsOptions {
            path: temp.path().join("SESSION.txt"),
            max_lines: 2,
            clear_on_start: false,
            clear_on_stop: false,
        };
        let result = validate_no_sink_path_collisions(Some(&save), Some(&obs), None);
        if directory_is_case_insensitive(temp.path()) {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("--save and --obs-text-file"));
        } else {
            result.unwrap();
        }
    }

    #[test]
    fn obs_sink_updates_only_on_final_or_revision_events() {
        let temp = tempfile::tempdir().unwrap();
        let obs_path = temp.path().join("obs.txt");
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.configure_prototype_sinks(
            Some(LiveObsOptions {
                path: obs_path.clone(),
                max_lines: 2,
                clear_on_start: false,
                clear_on_stop: false,
            }),
            None,
        );
        pipeline.start().unwrap();
        assert!(!obs_path.exists());

        let partial = RealtimeTranscriptPartial {
            utterance_id: TranscriptUtteranceId("utt_1".to_string()),
            segment_id: TranscriptSegmentId("seg_1".to_string()),
            revision: 1,
            text: "intermediate".to_string(),
            start_ms: 0,
            end_ms: 300,
            is_final: false,
            language: Some("en".to_string()),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        };
        let partial_envelope = pipeline
            .controller
            .transcript_event(RealtimeTranscriptEvent::Partial(partial), timestamp_now())
            .unwrap();
        pipeline.emit(partial_envelope).unwrap();
        assert!(!obs_path.exists());

        let final_first = RealtimeTranscriptFinal {
            utterance_id: TranscriptUtteranceId("utt_1".to_string()),
            segment_id: TranscriptSegmentId("seg_1".to_string()),
            revision: 1,
            text: "hello".to_string(),
            start_ms: 0,
            end_ms: 320,
            is_final: true,
            language: Some("en".to_string()),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        };
        let first_envelope = pipeline
            .controller
            .transcript_event(RealtimeTranscriptEvent::Final(final_first), timestamp_now())
            .unwrap();
        pipeline.emit(first_envelope).unwrap();

        let final_second = RealtimeTranscriptFinal {
            utterance_id: TranscriptUtteranceId("utt_2".to_string()),
            segment_id: TranscriptSegmentId("seg_1".to_string()),
            revision: 1,
            text: "world".to_string(),
            start_ms: 400,
            end_ms: 700,
            is_final: true,
            language: Some("en".to_string()),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        };
        let second_envelope = pipeline
            .controller
            .transcript_event(
                RealtimeTranscriptEvent::Final(final_second),
                timestamp_now(),
            )
            .unwrap();
        pipeline.emit(second_envelope).unwrap();

        let revision = RealtimeTranscriptRevision {
            utterance_id: TranscriptUtteranceId("utt_1".to_string()),
            segment_id: TranscriptSegmentId("seg_1".to_string()),
            revises_event_id: None,
            revision: 2,
            text: "hello there".to_string(),
            start_ms: 0,
            end_ms: 320,
            is_final: true,
            reason: "post_final_correction".to_string(),
            language: Some("en".to_string()),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        };
        let revision_envelope = pipeline
            .controller
            .transcript_event(RealtimeTranscriptEvent::Revision(revision), timestamp_now())
            .unwrap();
        pipeline.emit(revision_envelope).unwrap();

        pipeline.stop_obs_sink_worker();
        assert_eventually_file_equals(&obs_path, "hello there\nworld\n");
    }

    #[test]
    fn obs_sink_write_failures_do_not_abort_live_events() {
        let temp = tempfile::tempdir().unwrap();
        let obs_path = temp.path().join("obs-dir");
        std::fs::create_dir_all(&obs_path).unwrap();
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.configure_prototype_sinks(
            Some(LiveObsOptions {
                path: obs_path,
                max_lines: 2,
                clear_on_start: false,
                clear_on_stop: false,
            }),
            None,
        );
        pipeline.start().unwrap();

        let final_event = RealtimeTranscriptFinal {
            utterance_id: TranscriptUtteranceId("utt_1".to_string()),
            segment_id: TranscriptSegmentId("seg_1".to_string()),
            revision: 1,
            text: "hello".to_string(),
            start_ms: 0,
            end_ms: 320,
            is_final: true,
            language: Some("en".to_string()),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: Vec::new(),
        };
        let envelope = pipeline
            .controller
            .transcript_event(RealtimeTranscriptEvent::Final(final_event), timestamp_now())
            .unwrap();
        pipeline.emit(envelope).unwrap();
        assert_eq!(pipeline.history.entries().len(), 1);
    }

    #[test]
    fn obs_sink_truncates_to_max_lines() {
        let temp = tempfile::tempdir().unwrap();
        let obs_path = temp.path().join("obs.txt");
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.configure_prototype_sinks(
            Some(LiveObsOptions {
                path: obs_path.clone(),
                max_lines: 2,
                clear_on_start: false,
                clear_on_stop: false,
            }),
            None,
        );
        pipeline.start().unwrap();

        for (idx, text) in ["line one", "line two", "line three"]
            .into_iter()
            .enumerate()
        {
            let event = RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId(format!("utt_{}", idx + 1)),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: text.to_string(),
                start_ms: (idx as u64) * 100,
                end_ms: (idx as u64) * 100 + 80,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            };
            let envelope = pipeline
                .controller
                .transcript_event(RealtimeTranscriptEvent::Final(event), timestamp_now())
                .unwrap();
            pipeline.emit(envelope).unwrap();
        }
        pipeline.stop_obs_sink_worker();
        assert_eventually_file_equals(&obs_path, "line two\nline three\n");
    }

    #[test]
    fn obs_clear_flags_apply_on_start_and_stop() {
        let temp = tempfile::tempdir().unwrap();
        let obs_path = temp.path().join("obs.txt");
        std::fs::write(&obs_path, "stale\n").unwrap();
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.configure_prototype_sinks(
            Some(LiveObsOptions {
                path: obs_path.clone(),
                max_lines: 2,
                clear_on_start: true,
                clear_on_stop: true,
            }),
            None,
        );
        pipeline.start().unwrap();
        assert_eq!(std::fs::read_to_string(&obs_path).unwrap(), "");

        let mut worker = mock_transcription_worker();
        pipeline.shutdown(0, &mut worker, false).unwrap();
        assert_eq!(std::fs::read_to_string(obs_path).unwrap(), "");
    }

    #[test]
    fn obs_clear_on_start_failure_happens_before_session_events() {
        let temp = tempfile::tempdir().unwrap();
        let obs_path = temp.path().join("obs-dir");
        std::fs::create_dir_all(&obs_path).unwrap();
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.configure_prototype_sinks(
            Some(LiveObsOptions {
                path: obs_path,
                max_lines: 2,
                clear_on_start: true,
                clear_on_stop: false,
            }),
            None,
        );
        let error = pipeline.start().unwrap_err().to_string();
        assert!(error.contains("Could not clear OBS text file on start"));
        assert!(pipeline.emitted_events.is_empty());
    }

    #[test]
    fn obs_clear_on_stop_failure_does_not_fail_shutdown() {
        let temp = tempfile::tempdir().unwrap();
        let obs_path = temp.path().join("obs-dir");
        std::fs::create_dir_all(&obs_path).unwrap();
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.configure_prototype_sinks(
            Some(LiveObsOptions {
                path: obs_path,
                max_lines: 2,
                clear_on_start: false,
                clear_on_stop: true,
            }),
            None,
        );
        pipeline.start().unwrap();
        let mut worker = mock_transcription_worker();
        pipeline.shutdown(0, &mut worker, false).unwrap();
    }

    #[test]
    fn markdown_note_supports_append_and_title_suggestion() {
        let temp = tempfile::tempdir().unwrap();
        let note_path = temp.path().join("session.md");
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "hello world from openasr".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: note_path.clone(),
                append: false,
                title: None,
                suggest_title: true,
                created_at: "2026-05-10T00:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        pipeline.persist_markdown_note_if_requested().unwrap();
        let first = std::fs::read_to_string(&note_path).unwrap();
        assert!(first.contains("# hello world from openasr"));
        assert!(first.contains("- created_at: 2026-05-10T00:00:00.000Z"));
        assert!(first.contains("- source: microphone"));
        assert!(first.contains("- model: whisper-large-v3-turbo"));
        assert!(first.contains("- backend: mock"));
        assert!(first.contains("## Timeline"));
        assert!(first.contains("- 0-320 ms | seg_1 | rev 1 | hello world from openasr"));

        std::fs::write(&note_path, first.trim_end_matches('\n')).unwrap();

        let mut append_pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        append_pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_2".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "second session".to_string(),
                start_ms: 400,
                end_ms: 700,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_2".to_string()),
            Some(2),
        );
        append_pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: note_path.clone(),
                append: true,
                title: Some("Explicit Title".to_string()),
                suggest_title: false,
                created_at: "2026-05-10T01:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        append_pipeline
            .persist_markdown_note_if_requested()
            .unwrap();
        let appended = std::fs::read_to_string(note_path).unwrap();
        assert!(appended.contains("# hello world from openasr"));
        assert!(appended.contains("# Explicit Title"));
        assert!(appended.contains("\n\n# Explicit Title\n"));

        let append_only_path = temp.path().join("append-only.md");
        let mut append_only_pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        append_only_pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_3".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "append first write".to_string(),
                start_ms: 800,
                end_ms: 900,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_3".to_string()),
            Some(3),
        );
        append_only_pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: append_only_path.clone(),
                append: true,
                title: None,
                suggest_title: false,
                created_at: "2026-05-10T02:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        append_only_pipeline
            .persist_markdown_note_if_requested()
            .unwrap();
        let append_only = std::fs::read_to_string(append_only_path).unwrap();
        assert!(append_only.contains("# OpenASR Live Session"));
        assert!(append_only.contains("append first write"));
    }

    #[test]
    fn markdown_timeline_normalizes_multiline_entry_text() {
        let temp = tempfile::tempdir().unwrap();
        let note_path = temp.path().join("session.md");
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "hello\nfrom\topenasr".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: note_path.clone(),
                append: false,
                title: Some("Session".to_string()),
                suggest_title: false,
                created_at: "2026-05-10T00:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        pipeline.persist_markdown_note_if_requested().unwrap();
        let rendered = std::fs::read_to_string(note_path).unwrap();
        assert!(rendered.contains("- 0-320 ms | seg_1 | rev 1 | hello from openasr"));
    }

    #[test]
    fn markdown_note_escapes_transcript_markdown_syntax() {
        let temp = tempfile::tempdir().unwrap();
        let note_path = temp.path().join("session.md");
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "- todo > quote # heading `code`".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: note_path.clone(),
                append: false,
                title: Some("Session".to_string()),
                suggest_title: false,
                created_at: "2026-05-10T00:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        pipeline.persist_markdown_note_if_requested().unwrap();
        let rendered = std::fs::read_to_string(note_path).unwrap();
        assert!(rendered.contains("- \\- todo \\> quote \\# heading \\`code\\`"));
        assert!(rendered.contains("| \\- todo \\> quote \\# heading \\`code\\`"));
    }

    #[test]
    fn markdown_note_escapes_suggested_title_markdown_syntax() {
        let temp = tempfile::tempdir().unwrap();
        let note_path = temp.path().join("session.md");
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "[release](v1) now".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: note_path.clone(),
                append: false,
                title: None,
                suggest_title: true,
                created_at: "2026-05-10T00:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        pipeline.persist_markdown_note_if_requested().unwrap();
        let rendered = std::fs::read_to_string(note_path).unwrap();
        assert!(rendered.contains("# \\[release\\]\\(v1\\) now"));
    }

    #[test]
    fn partial_history_save_uses_sidecar_path_without_overwriting_target() {
        let temp = tempfile::tempdir().unwrap();
        let save_path = temp.path().join("live.md");
        std::fs::write(&save_path, "keep-existing\n").unwrap();
        let save_options = Some(LiveSaveOptions {
            path: save_path.clone(),
            format: RealtimeExportFormat::Markdown,
            post_processor: RealtimePostProcessor::default(),
        });
        let mut pipeline = LivePipeline::new(
            test_live_config(),
            LiveOutputFormat::Jsonl,
            None,
            save_options,
        )
        .unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "hello".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        pipeline.persist_partial_history_if_requested().unwrap();
        let partial_path = PathBuf::from(format!("{}.partial", save_path.display()));
        assert_eq!(
            std::fs::read_to_string(save_path).unwrap(),
            "keep-existing\n"
        );
        let partial = std::fs::read_to_string(partial_path).unwrap();
        assert!(partial.contains("hello"));
    }

    #[test]
    fn partial_history_save_refuses_to_overwrite_existing_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let save_path = temp.path().join("live.md");
        let partial_path = PathBuf::from(format!("{}.partial", save_path.display()));
        std::fs::write(&partial_path, "existing-partial\n").unwrap();
        let save_options = Some(LiveSaveOptions {
            path: save_path,
            format: RealtimeExportFormat::Markdown,
            post_processor: RealtimePostProcessor::default(),
        });
        let mut pipeline = LivePipeline::new(
            test_live_config(),
            LiveOutputFormat::Jsonl,
            None,
            save_options,
        )
        .unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "hello".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        let error = pipeline.persist_partial_history_if_requested().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Refusing to overwrite implicit partial live transcript sidecar")
        );
        assert_eq!(
            std::fs::read_to_string(partial_path).unwrap(),
            "existing-partial\n"
        );
    }

    #[test]
    fn shutdown_save_failure_falls_back_to_partial_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let save_path = temp.path().join("live.md");
        let note_path = temp.path().join("session.md");
        std::fs::create_dir_all(&save_path).unwrap();
        let save_options = Some(LiveSaveOptions {
            path: save_path.clone(),
            format: RealtimeExportFormat::Markdown,
            post_processor: RealtimePostProcessor::default(),
        });
        let mut pipeline = LivePipeline::new(
            test_live_config(),
            LiveOutputFormat::Jsonl,
            None,
            save_options,
        )
        .unwrap();
        pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: note_path.clone(),
                append: false,
                title: Some("Session".to_string()),
                suggest_title: false,
                created_at: "2026-05-10T00:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        let mut worker = mock_transcription_worker();
        pipeline.start().unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "hello".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        let error = pipeline
            .shutdown(0, &mut worker, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Could not save live transcript history"));
        let partial_path = PathBuf::from(format!("{}.partial", save_path.display()));
        let partial = std::fs::read_to_string(partial_path).unwrap();
        assert!(partial.contains("hello"));
        assert!(
            std::fs::read_to_string(note_path)
                .unwrap()
                .contains("hello")
        );
    }

    #[test]
    fn subtitle_save_failure_falls_back_to_text_partial_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let save_path = temp.path().join("live.srt");
        let save_options = Some(LiveSaveOptions {
            path: save_path.clone(),
            format: RealtimeExportFormat::Srt,
            post_processor: RealtimePostProcessor::default(),
        });
        let mut pipeline = LivePipeline::new(
            test_live_config(),
            LiveOutputFormat::Jsonl,
            None,
            save_options,
        )
        .unwrap();
        let mut worker = mock_transcription_worker();
        pipeline.start().unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "hello".to_string(),
                start_ms: 100,
                end_ms: 100,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        let error = pipeline
            .shutdown(0, &mut worker, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Could not render live transcript history"));
        let partial_path = PathBuf::from(format!("{}.partial", save_path.display()));
        let partial = std::fs::read_to_string(partial_path).unwrap();
        assert_eq!(partial, "hello\n");
    }

    #[test]
    fn shutdown_error_cleanup_persists_partial_sidecar_without_final_save() {
        let temp = tempfile::tempdir().unwrap();
        let save_path = temp.path().join("live.md");
        let note_path = temp.path().join("session.md");
        let save_options = Some(LiveSaveOptions {
            path: save_path.clone(),
            format: RealtimeExportFormat::Markdown,
            post_processor: RealtimePostProcessor::default(),
        });
        let mut pipeline = LivePipeline::new(
            test_live_config(),
            LiveOutputFormat::Jsonl,
            None,
            save_options,
        )
        .unwrap();
        pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: note_path.clone(),
                append: false,
                title: Some("Session".to_string()),
                suggest_title: false,
                created_at: "2026-05-10T00:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        let mut worker = mock_transcription_worker();
        pipeline.start().unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "hello".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        pipeline.shutdown(0, &mut worker, false).unwrap();
        let partial_path = PathBuf::from(format!("{}.partial", save_path.display()));
        let note_partial_path = PathBuf::from(format!("{}.partial", note_path.display()));
        assert!(!save_path.exists());
        assert!(
            std::fs::read_to_string(partial_path)
                .unwrap()
                .contains("hello")
        );
        assert!(!note_path.exists());
        assert!(
            std::fs::read_to_string(note_partial_path)
                .unwrap()
                .contains("hello")
        );
    }

    #[test]
    fn shutdown_error_writes_markdown_partial_sidecar_even_when_append_enabled() {
        let temp = tempfile::tempdir().unwrap();
        let note_path = temp.path().join("session.md");
        std::fs::write(&note_path, "existing\n").unwrap();
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.configure_prototype_sinks(
            None,
            Some(LiveMarkdownNoteOptions {
                path: note_path.clone(),
                append: true,
                title: Some("Session".to_string()),
                suggest_title: false,
                created_at: "2026-05-10T00:00:00.000Z".to_string(),
                source_label: "microphone".to_string(),
                model_id: "whisper-large-v3-turbo".to_string(),
                backend: "mock".to_string(),
            }),
        );
        let mut worker = mock_transcription_worker();
        pipeline.start().unwrap();
        pipeline.history.apply_final(
            &RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_1".to_string()),
                segment_id: TranscriptSegmentId("seg_1".to_string()),
                revision: 1,
                text: "hello".to_string(),
                start_ms: 0,
                end_ms: 320,
                is_final: true,
                language: Some("en".to_string()),
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
                words: Vec::new(),
            },
            Some("evt_1".to_string()),
            Some(1),
        );
        pipeline.shutdown(0, &mut worker, false).unwrap();
        let note_partial_path = PathBuf::from(format!("{}.partial", note_path.display()));
        assert_eq!(std::fs::read_to_string(&note_path).unwrap(), "existing\n");
        assert!(
            std::fs::read_to_string(note_partial_path)
                .unwrap()
                .contains("hello")
        );
    }

    #[test]
    fn normalizer_rejects_invalid_config_and_bad_interleaving() {
        assert!(
            LiveAudioNormalizer::new(0, 1, 20)
                .unwrap_err()
                .to_string()
                .contains("sample rate")
        );
        assert!(
            LiveAudioNormalizer::new(16_000, 0, 20)
                .unwrap_err()
                .to_string()
                .contains("channel")
        );
        assert!(
            LiveAudioNormalizer::new(16_000, 1, 25)
                .unwrap_err()
                .to_string()
                .contains("frame duration")
        );

        let mut normalizer = LiveAudioNormalizer::new(16_000, 2, 20).unwrap();
        assert!(
            normalizer
                .push_f32_interleaved(&[0.0, 0.1, 0.2])
                .unwrap_err()
                .to_string()
                .contains("not divisible")
        );
    }

    #[test]
    fn normalizer_passes_through_16khz_mono_and_splits_frames() {
        let mut normalizer = LiveAudioNormalizer::new(16_000, 1, 20).unwrap();
        let samples = vec![0.5_f32; 640];
        let frames = normalizer.push_f32_interleaved(&samples).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].seq, 1);
        assert_eq!(frames[0].start_ms, 0);
        assert_eq!(frames[0].samples()[0], 16384);
        assert_eq!(frames[1].seq, 2);
        assert_eq!(frames[1].start_ms, 20);
    }

    #[test]
    fn normalizer_resamples_48khz_mono_to_16khz_frames() {
        let mut normalizer = LiveAudioNormalizer::new(48_000, 1, 20).unwrap();
        let samples = vec![0.25_f32; 960];
        let frames = normalizer.push_f32_interleaved(&samples).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].sample_count(), 320);
        assert!(frames[0].samples().iter().all(|sample| *sample == 8192));
    }

    #[test]
    fn normalizer_downmixes_stereo_and_converts_i16_u16() {
        let mut i16_norm = LiveAudioNormalizer::new(16_000, 2, 20).unwrap();
        let i16_samples = (0..320)
            .flat_map(|_| [32767_i16, -32768_i16])
            .collect::<Vec<_>>();
        let frames = i16_norm.push_i16_interleaved(&i16_samples).unwrap();
        assert_eq!(frames.len(), 1);
        assert!(frames[0].samples()[0].abs() <= 1);

        let mut u16_norm = LiveAudioNormalizer::new(16_000, 1, 20).unwrap();
        let u16_samples = vec![u16::MAX; 320];
        let frames = u16_norm.push_u16_interleaved(&u16_samples).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].samples()[0], 32766);
    }

    #[test]
    fn normalizer_carries_partial_samples_across_chunks() {
        let mut normalizer = LiveAudioNormalizer::new(16_000, 1, 20).unwrap();
        assert!(
            normalizer
                .push_f32_interleaved(&vec![0.1_f32; 200])
                .unwrap()
                .is_empty()
        );
        let frames = normalizer
            .push_f32_interleaved(&vec![0.1_f32; 120])
            .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].sample_count(), 320);
    }

    #[test]
    fn wav_writer_sets_header_and_data() {
        let mut bytes = Vec::new();
        write_pcm16_mono_16khz_wav(&mut bytes, &[1, -2]).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1);
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1);
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            16_000
        );
        assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16);
        assert_eq!(
            u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]),
            4
        );
        assert_eq!(i16::from_le_bytes([bytes[44], bytes[45]]), 1);
        assert_eq!(i16::from_le_bytes([bytes[46], bytes[47]]), -2);
    }

    #[test]
    fn timestamp_formatter_uses_rfc3339_utc_shape() {
        assert_eq!(format_unix_millis(0, 0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            format_unix_millis(1_766_793_600, 123),
            "2025-12-27T00:00:00.123Z"
        );
    }

    #[test]
    fn temp_wav_is_removed_after_drop() {
        let utterance = BufferedUtterance {
            utterance_id: openasr_core::TranscriptUtteranceId("utt_1".to_string()),
            start_ms: 0,
            end_ms: 20,
            frames: vec![frame(1, 0, 1000)],
            reason: RealtimeUtteranceEndReason::VadStop,
        };
        let file = write_temp_utterance_wav(&utterance).unwrap();
        let path = file.path().to_path_buf();
        assert!(path.exists());
        drop(file);
        assert!(!path.exists());
    }

    #[test]
    fn fake_pipeline_emits_vad_and_final_without_fake_fields() {
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        let mut worker = mock_transcription_worker();
        pipeline.start().unwrap();
        for (index, sample) in [0, 2000, 2000, 0, 0].into_iter().enumerate() {
            pipeline
                .process_frame(
                    frame(index as u64 + 1, index as u64 * 20, sample),
                    &mut worker,
                )
                .unwrap();
        }
        pipeline.shutdown(100, &mut worker, true).unwrap();
        assert_eq!(pipeline.completed_utterances, 1);

        let event_types = pipeline
            .emitted_events
            .iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        assert_eq!(
            event_types,
            vec![
                "session.created",
                "session.configured",
                "audio.input.started",
                "vad.speech_started",
                "vad.speech_stopped",
                "transcript.final",
                "audio.input.stopped",
                "session.closed",
            ]
        );
        for (expected_seq, event) in pipeline.emitted_events.iter().enumerate() {
            assert_eq!(event.seq, expected_seq as u64 + 1);
        }

        let configured = serde_json::to_value(&pipeline.emitted_events[1]).unwrap();
        assert_eq!(configured["partial_results"], true);

        for event in &pipeline.emitted_events {
            let json = serde_json::to_value(event).unwrap();
            for forbidden in ["speaker", "words", "word", "confidence", "stability"] {
                assert!(
                    json.get(forbidden).is_none(),
                    "unexpected field {forbidden} in {json}"
                );
            }
            assert_ne!(json["type"], "transcript.partial");
        }
    }

    #[test]
    fn duplicate_partial_text_is_suppressed() {
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.start().unwrap();

        let base = LiveTranscriptionSuccess {
            utterance_id: TranscriptUtteranceId("utt_dup".to_string()),
            start_ms: 0,
            end_ms: 200,
            segment_id: TranscriptSegmentId("utt_dup_seg_000001".to_string()),
            text: "hello world".to_string(),
            partial: true,
            revision: 1,
            generation: 0,
        };
        pipeline
            .apply_transcription_result(LiveTranscriptionResult::Final(base), None)
            .unwrap();

        let duplicate = LiveTranscriptionSuccess {
            utterance_id: TranscriptUtteranceId("utt_dup".to_string()),
            start_ms: 0,
            end_ms: 260,
            segment_id: TranscriptSegmentId("utt_dup_seg_000001".to_string()),
            text: "hello world".to_string(),
            partial: true,
            revision: 2,
            generation: 0,
        };
        pipeline
            .apply_transcription_result(LiveTranscriptionResult::Final(duplicate), None)
            .unwrap();

        let partial_count = pipeline
            .emitted_events
            .iter()
            .filter(|event| event.event_type == "transcript.partial")
            .count();
        assert_eq!(partial_count, 1);
    }

    #[test]
    fn partial_rollback_is_suppressed_when_new_text_is_shorter_prefix() {
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.start().unwrap();

        let first = LiveTranscriptionSuccess {
            utterance_id: TranscriptUtteranceId("utt_rb".to_string()),
            start_ms: 0,
            end_ms: 200,
            segment_id: TranscriptSegmentId("utt_rb_seg_000001".to_string()),
            text: "hello world from openasr".to_string(),
            partial: true,
            revision: 1,
            generation: 0,
        };
        pipeline
            .apply_transcription_result(LiveTranscriptionResult::Final(first), None)
            .unwrap();

        let rollback = LiveTranscriptionSuccess {
            utterance_id: TranscriptUtteranceId("utt_rb".to_string()),
            start_ms: 0,
            end_ms: 260,
            segment_id: TranscriptSegmentId("utt_rb_seg_000001".to_string()),
            text: "hello world".to_string(),
            partial: true,
            revision: 2,
            generation: 0,
        };
        pipeline
            .apply_transcription_result(LiveTranscriptionResult::Final(rollback), None)
            .unwrap();

        let partial_count = pipeline
            .emitted_events
            .iter()
            .filter(|event| event.event_type == "transcript.partial")
            .count();
        assert_eq!(partial_count, 1);
    }

    #[test]
    fn partial_dispatch_keeps_one_in_flight_and_only_the_latest_pending() {
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        let mut worker = mock_transcription_worker();
        let utterance_id = TranscriptUtteranceId("utt_flight".to_string());
        let first = buffered_utterance("utt_flight", 1, 0, 1000);
        let latest = buffered_utterance("utt_flight", 2, 20, 2000);

        pipeline
            .partial_flights
            .entry(utterance_id.clone())
            .or_default()
            .pending_latest = Some(first);
        pipeline
            .queue_next_partial(&utterance_id, 300, &mut worker)
            .unwrap();

        assert_eq!(worker.pending_jobs, 1);
        let flight = pipeline.partial_flights.get_mut(&utterance_id).unwrap();
        assert!(flight.in_flight);
        assert!(flight.pending_latest.is_none());
        assert_eq!(flight.next_revision, 1);
        flight.pending_latest = Some(latest.clone());

        pipeline
            .queue_next_partial(&utterance_id, 600, &mut worker)
            .unwrap();

        let flight = pipeline.partial_flights.get(&utterance_id).unwrap();
        assert_eq!(worker.pending_jobs, 1);
        assert!(flight.in_flight);
        assert_eq!(
            flight.pending_latest.as_ref().unwrap().end_ms,
            latest.end_ms
        );
    }

    #[test]
    fn late_partial_generation_cannot_override_final_text() {
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.start().unwrap();
        let utterance_id = TranscriptUtteranceId("utt_final_guard".to_string());
        let segment_id = TranscriptSegmentId("utt_final_guard_seg_000001".to_string());

        pipeline
            .apply_transcription_result(
                LiveTranscriptionResult::Final(LiveTranscriptionSuccess {
                    utterance_id: utterance_id.clone(),
                    start_ms: 0,
                    end_ms: 200,
                    segment_id: segment_id.clone(),
                    text: "hello wor".to_string(),
                    partial: true,
                    revision: 1,
                    generation: 0,
                }),
                None,
            )
            .unwrap();
        pipeline
            .apply_transcription_result(
                LiveTranscriptionResult::Final(LiveTranscriptionSuccess {
                    utterance_id: utterance_id.clone(),
                    start_ms: 0,
                    end_ms: 260,
                    segment_id: segment_id.clone(),
                    text: "hello world".to_string(),
                    partial: false,
                    revision: 2,
                    generation: 1,
                }),
                None,
            )
            .unwrap();
        pipeline
            .apply_transcription_result(
                LiveTranscriptionResult::Final(LiveTranscriptionSuccess {
                    utterance_id,
                    start_ms: 0,
                    end_ms: 300,
                    segment_id,
                    text: "hello wrong stale partial".to_string(),
                    partial: true,
                    revision: 3,
                    generation: 0,
                }),
                None,
            )
            .unwrap();

        let transcript_events = pipeline
            .emitted_events
            .iter()
            .filter(|event| {
                matches!(
                    event.event_type,
                    "transcript.partial" | "transcript.final" | "transcript.revision"
                )
            })
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        assert_eq!(
            transcript_events,
            vec!["transcript.partial", "transcript.final"]
        );
        assert_eq!(pipeline.partial_emitted_count, 1);
        assert_eq!(pipeline.final_emitted_count, 1);
    }

    #[test]
    fn backend_error_still_emits_terminal_events() {
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        let mut worker = LiveTranscriptionWorker::spawn(BackendKind::Native, None);
        pipeline.start().unwrap();
        for (index, sample) in [0, 2000, 2000, 0, 0].into_iter().enumerate() {
            pipeline
                .process_frame(
                    frame(index as u64 + 1, index as u64 * 20, sample),
                    &mut worker,
                )
                .unwrap();
        }
        let error = pipeline
            .shutdown(100, &mut worker, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("Could not transcribe completed live utterance"));

        let event_types = pipeline
            .emitted_events
            .iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        let error_index = event_types
            .iter()
            .position(|event| *event == "error")
            .unwrap();
        let stopped_index = event_types
            .iter()
            .position(|event| *event == "audio.input.stopped")
            .unwrap();
        let closed_index = event_types
            .iter()
            .position(|event| *event == "session.closed")
            .unwrap();
        assert!(error_index < stopped_index);
        assert!(stopped_index < closed_index);
    }

    #[test]
    fn capture_run_overflow_emits_backpressure_and_closes() {
        let (_sender, receiver) = mpsc::sync_channel(1);
        let overflowed = Arc::new(AtomicBool::new(true));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, None, None).unwrap();
        pipeline.start().unwrap();
        let mut run = LiveCaptureRun {
            receiver,
            normalizer: LiveAudioNormalizer::new(16_000, 1, 20).unwrap(),
            overflowed,
            started_at: Instant::now(),
            max_seconds: None,
            max_utterances: None,
            stop_requested,
            transcription_worker: mock_transcription_worker(),
        };
        let error = run.run(&mut pipeline).unwrap_err().to_string();
        assert!(error.contains("queue overflowed"));

        let event_types = pipeline
            .emitted_events
            .iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        assert!(event_types.contains(&"error"));
        assert!(event_types.contains(&"audio.input.stopped"));
        assert!(event_types.contains(&"session.closed"));
        let error_event = pipeline
            .emitted_events
            .iter()
            .find(|event| event.event_type == "error")
            .unwrap();
        let json = serde_json::to_value(error_event).unwrap();
        assert_eq!(json["code"], "backpressure_timeout");
        assert_eq!(json["recoverable"], false);
    }

    #[test]
    fn capture_run_stops_after_max_utterances_and_closes_in_order() {
        let (sender, receiver) = mpsc::sync_channel(16);
        sender.send(CaptureChunk::I16(vec![0; 320])).unwrap();
        sender.send(CaptureChunk::I16(vec![2000; 320])).unwrap();
        sender.send(CaptureChunk::I16(vec![2000; 320])).unwrap();
        sender.send(CaptureChunk::I16(vec![0; 320])).unwrap();
        sender.send(CaptureChunk::I16(vec![0; 320])).unwrap();
        sender.send(CaptureChunk::I16(vec![2000; 320])).unwrap();
        sender.send(CaptureChunk::I16(vec![2000; 320])).unwrap();
        sender.send(CaptureChunk::I16(vec![0; 320])).unwrap();
        sender.send(CaptureChunk::I16(vec![0; 320])).unwrap();
        drop(sender);

        let mut pipeline =
            LivePipeline::new(test_live_config(), LiveOutputFormat::Jsonl, Some(1), None).unwrap();
        pipeline.start().unwrap();
        let mut run = LiveCaptureRun {
            receiver,
            normalizer: LiveAudioNormalizer::new(16_000, 1, 20).unwrap(),
            overflowed: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
            max_seconds: None,
            max_utterances: Some(1),
            stop_requested: Arc::new(AtomicBool::new(false)),
            transcription_worker: mock_transcription_worker(),
        };
        run.run(&mut pipeline).unwrap();
        assert_eq!(pipeline.accepted_utterances, 1);
        assert_eq!(pipeline.completed_utterances, 1);

        let event_types = pipeline
            .emitted_events
            .iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>();
        let final_index = event_types
            .iter()
            .position(|event| *event == "transcript.final")
            .unwrap();
        let stopped_index = event_types
            .iter()
            .position(|event| *event == "audio.input.stopped")
            .unwrap();
        let closed_index = event_types
            .iter()
            .position(|event| *event == "session.closed")
            .unwrap();
        assert!(final_index < stopped_index);
        assert!(stopped_index < closed_index);
    }

    #[test]
    fn buffer_overflow_returns_typed_error() {
        let config = LivePipelineConfig {
            model_id: "whisper-large-v3-turbo".to_string(),
            model_pack_path: None,
            diarize: false,
            vad: VadConfig {
                frame_duration_ms: 20,
                speech_start_ms: 20,
                speech_stop_ms: 200,
                pre_roll_ms: 0,
                max_utterance_ms: None,
                no_speech_timeout_ms: None,
                mode: VadMode::Energy,
                energy_threshold: 0.02,
            },
            buffer: RealtimeBufferConfig {
                frame_duration_ms: 20,
                pre_roll_ms: 0,
                max_buffered_frames: 1,
                max_buffered_samples: 320,
            },
            partial_interval_ms: DEFAULT_STREAMING_PARTIAL_INTERVAL_MS,
            partial_window_ms: DEFAULT_STREAMING_PARTIAL_WINDOW_MS,
        };
        let mut pipeline = LivePipeline::new(config, LiveOutputFormat::Jsonl, None, None).unwrap();
        let mut worker = mock_transcription_worker();
        pipeline.start().unwrap();
        pipeline
            .process_frame(frame(1, 0, 2000), &mut worker)
            .unwrap();
        let error = pipeline
            .process_frame(frame(2, 20, 2000), &mut worker)
            .unwrap_err()
            .to_string();
        assert!(error.contains("realtime audio buffer reached capacity"));
    }

    #[test]
    fn shutdown_flushes_active_utterance() {
        let config = LivePipelineConfig {
            model_id: "whisper-large-v3-turbo".to_string(),
            model_pack_path: None,
            diarize: false,
            vad: VadConfig {
                frame_duration_ms: 20,
                speech_start_ms: 20,
                speech_stop_ms: 200,
                pre_roll_ms: 0,
                max_utterance_ms: None,
                no_speech_timeout_ms: None,
                mode: VadMode::Energy,
                energy_threshold: 0.02,
            },
            buffer: RealtimeBufferConfig {
                frame_duration_ms: 20,
                pre_roll_ms: 0,
                max_buffered_frames: 20,
                max_buffered_samples: 10_000,
            },
            partial_interval_ms: DEFAULT_STREAMING_PARTIAL_INTERVAL_MS,
            partial_window_ms: DEFAULT_STREAMING_PARTIAL_WINDOW_MS,
        };
        let mut pipeline = LivePipeline::new(config, LiveOutputFormat::Text, None, None).unwrap();
        let mut worker = mock_transcription_worker();
        pipeline.start().unwrap();
        pipeline
            .process_frame(frame(1, 0, 2000), &mut worker)
            .unwrap();
        pipeline.shutdown(20, &mut worker, true).unwrap();
        assert_eq!(pipeline.completed_utterances, 1);
    }
}
