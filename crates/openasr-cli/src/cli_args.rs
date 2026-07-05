use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use openasr_core::{BackendKind, BenchmarkFormat, ResponseFormat, TranscriptionTask};

use crate::{
    live, parse_backend_kind, parse_benchmark_format, parse_response_format,
    parse_transcription_task,
};

#[derive(Debug, Default, Clone)]
pub(crate) struct RuntimePathOverrides {
    pub(crate) ffmpeg_bin: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub(crate) struct TranscribeCommandOptions<'a> {
    pub(crate) inputs: &'a [PathBuf],
    pub(crate) formats: &'a [ResponseFormat],
    pub(crate) model: Option<&'a str>,
    pub(crate) backend_kind: Option<BackendKind>,
    pub(crate) runtime_paths: RuntimePathOverrides,
    pub(crate) diarize: bool,
    pub(crate) speakers: Option<u8>,
    pub(crate) word_timestamps: bool,
    pub(crate) model_pack: Option<&'a Path>,
    /// OADP Phase 0 `.oadp` adapter pack; plumbed through the transcription
    /// request (never the process environment — workers are already running).
    pub(crate) adapter: Option<&'a Path>,
    pub(crate) output: Option<&'a Path>,
    pub(crate) continue_on_error: bool,
    pub(crate) benchmark: bool,
    pub(crate) longform: NativeLongFormCliOptions,
    pub(crate) phrase_bias: PhraseBiasCliOptions,
    pub(crate) language: Option<String>,
    pub(crate) task: Option<TranscriptionTask>,
    /// Non-interactive consent for the auto-pull of a missing model.
    pub(crate) consent: crate::consent::PullConsent,
}

#[derive(Debug, Clone)]
pub(crate) struct BenchSuiteCommandOptions<'a> {
    pub(crate) config: &'a Path,
    pub(crate) baseline: Option<&'a Path>,
    pub(crate) write_baseline: Option<&'a Path>,
    pub(crate) format: BenchmarkFormat,
    pub(crate) family: Option<&'a str>,
    pub(crate) runs: usize,
    pub(crate) run_single_entry: Option<&'a str>,
    pub(crate) runtime_paths: RuntimePathOverrides,
}

#[derive(Debug, Clone)]
pub(crate) struct BatchRunContext<'a> {
    pub(crate) output_dir: &'a Path,
    pub(crate) formats: &'a [ResponseFormat],
    pub(crate) model_id: &'a str,
    pub(crate) model_pack_path: Option<PathBuf>,
    pub(crate) backend_kind: BackendKind,
    pub(crate) ffmpeg_bin: Option<PathBuf>,
    pub(crate) longform: Option<openasr_core::LongFormOptions>,
    pub(crate) diarize: bool,
    pub(crate) speakers: Option<u8>,
    pub(crate) language: Option<String>,
    pub(crate) task: Option<TranscriptionTask>,
}

#[derive(Debug, Clone)]
pub(crate) struct PullCommandOptions<'a> {
    pub(crate) reference: &'a str,
    pub(crate) quant: Option<&'a str>,
    pub(crate) size: Option<&'a str>,
    pub(crate) catalog_url: Option<&'a str>,
    pub(crate) source: Option<&'a str>,
    pub(crate) accept_license: bool,
    pub(crate) from: Option<&'a Path>,
}

#[derive(Debug, Parser)]
#[command(name = "openasr")]
#[command(about = "Local-first speech-to-text -- no cloud, no telemetry, fail-closed by design")]
#[command(version)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Clone, PartialEq, Args)]
pub(crate) struct NativeLongFormCliOptions {
    /// Native longform segmentation mode for ggml local runtime execution.
    #[arg(long, hide = true)]
    pub(crate) segment_mode: Option<NativeSegmentMode>,
    /// Native longform chunk length in seconds.
    #[arg(long, hide = true)]
    pub(crate) chunk_seconds: Option<f64>,
    /// Native longform overlap between adjacent chunks in seconds.
    #[arg(long, default_value_t = 0.5, hide = true)]
    pub(crate) segment_overlap_seconds: f64,
    /// Native longform silence threshold in dBFS for energy-aware splitting/suppression.
    #[arg(long, default_value_t = -38.0, hide = true)]
    pub(crate) vad_threshold_db: f32,
    /// Native longform VAD minimum silence duration that ends a segment.
    #[arg(long, default_value_t = 450, hide = true)]
    pub(crate) vad_min_silence_ms: usize,
    /// Native longform context padding around each segment.
    #[arg(long, default_value_t = 250, hide = true)]
    pub(crate) vad_padding_ms: usize,
    /// Native longform minimum segment duration before padding.
    #[arg(long, default_value_t = 1.0, hide = true)]
    pub(crate) min_segment_seconds: f64,
    /// Skip whole longform chunks whose in-window audio is effectively silent.
    #[arg(long, default_value_t = false, hide = true)]
    pub(crate) suppress_silent_slices: bool,
}

impl Default for NativeLongFormCliOptions {
    fn default() -> Self {
        Self {
            segment_mode: None,
            chunk_seconds: None,
            segment_overlap_seconds: 0.5,
            vad_threshold_db: -38.0,
            vad_min_silence_ms: 450,
            vad_padding_ms: 250,
            min_segment_seconds: 1.0,
            suppress_silent_slices: false,
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Args)]
pub(crate) struct PhraseBiasCliOptions {
    /// Bias transcription toward this phrase. Repeat for multiple hotwords.
    #[arg(long = "hotword", value_name = "PHRASE")]
    pub(crate) hotwords: Vec<String>,
    /// Base boost for each --hotword phrase. Defaults to 5.0 when --hotword is
    /// present. Positive favors the phrase; a negative value suppresses it
    /// (anti-context). Applied as-is to a phrase's first token and scaled up
    /// mid-phrase with matched depth; every applied value is capped at 20.0.
    #[arg(long = "hotword-boost", value_name = "BOOST")]
    pub(crate) hotword_boost: Option<f32>,
}

#[derive(Debug, Default, Clone, PartialEq, Args)]
pub(crate) struct LanguageTaskCliOptions {
    /// Source language hint (e.g. en, fr, zh). Use `auto` or omit to let the
    /// model detect the language.
    #[arg(long, short = 'l', value_name = "LANG")]
    pub(crate) language: Option<String>,
    /// Speech task: transcribe (keep the source language) or translate (to English).
    /// Whisper-only; other families reject translate / a non-default language.
    #[arg(long, value_parser = parse_transcription_task, value_name = "TASK")]
    pub(crate) task: Option<TranscriptionTask>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// List installed model packs.
    List,
    /// Search the model catalog for models you can pull.
    Search {
        /// Optional name or family filter.
        query: Option<String>,
    },
    /// Download a local OpenASR model pack from the model catalog.
    Pull {
        /// Model reference in <id> or <id>:<quant> form, for example moonshine-tiny:q8.
        reference: String,
        /// Override the quant suffix or quant id, for example q8 or q8_0.
        #[arg(long)]
        quant: Option<String>,
        /// Disambiguate an alias by model size when needed.
        #[arg(long)]
        size: Option<String>,
        /// Override the model catalog URL or local catalog path.
        #[arg(long)]
        catalog_url: Option<String>,
        /// Download source: auto, hf, or hf-mirror.
        #[arg(long, value_parser = ["auto", "hf", "hf-mirror"])]
        source: Option<String>,
        /// Acknowledge the model license when the catalog requires it.
        #[arg(long)]
        accept_license: bool,
        /// Use an already downloaded local pack for gated/vendor flows.
        #[arg(long)]
        from: Option<PathBuf>,
    },
    /// Remove an installed model pack.
    Rm {
        /// Installed model id (optionally with a quant suffix).
        id: String,
    },
    /// Read and update saved OpenASR config.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Print local OpenASR environment diagnostics.
    Doctor,
    /// Verify a local OpenASR model pack (`.oasr`) via a ggml integrity probe.
    Verify {
        /// Path to a local `.oasr` pack file.
        path: PathBuf,
    },
    /// Show details for a model id (catalog card) or a local `.oasr` pack file.
    Show {
        /// A model id, or a path to a local `.oasr` pack.
        target: String,
    },
    /// Validate and inspect local OpenASR model packs (`.oasr`).
    ModelPack {
        #[command(subcommand)]
        command: ModelPackCommand,
    },
    /// Internal helper for sandboxed GGUF C parser probes.
    #[command(name = "__openasr-gguf-c-parser-probe", hide = true)]
    GgufCParserProbe {
        /// Runtime pack path to parse.
        path: PathBuf,
    },
    /// Internal helper for release catalog signature manifests.
    #[command(name = "__openasr-sign-catalog-manifest", hide = true)]
    SignCatalogManifest {
        /// Catalog JSON file to sign.
        catalog: PathBuf,
        /// Output catalog.signature.json path.
        #[arg(long)]
        out: PathBuf,
        /// Monotonic catalog epoch.
        #[arg(long)]
        epoch: u64,
        /// Override catalog_url from the catalog JSON.
        #[arg(long)]
        catalog_url: Option<String>,
        /// Signature key id.
        #[arg(long, default_value = "openasr-catalog-v1")]
        key_id: String,
        /// Print the derived public key for the env signing seed and exit.
        #[arg(long)]
        print_public_key: bool,
    },
    /// Internal helper: print the embedded bundled catalog's signature-verified
    /// fingerprint (sha256 + epoch) as a single JSON line. No network, no side
    /// effects. Used by packaging tooling to confirm a prebuilt sidecar's
    /// embedded catalog matches a copied catalog resource.
    #[command(name = "catalog-fingerprint", hide = true)]
    CatalogFingerprint,
    /// Transcribe one or more audio files (or directories of audio).
    #[command(visible_alias = "t")]
    Transcribe {
        /// Audio file(s) or directories. A single file prints to stdout (or
        /// `--output`); multiple inputs or a directory write one transcript per
        /// file into the `--output` directory.
        #[arg(required = true, num_args = 1.., value_name = "INPUTS")]
        inputs: Vec<PathBuf>,
        /// Output format(s): text, json, srt, vtt, verbose_json, markdown. Repeat
        /// `-f` to write several at once as sidecar files (next to the input, or
        /// in the `--output` directory).
        #[arg(long = "format", short = 'f', value_name = "FORMAT", default_value = "text", value_parser = parse_response_format)]
        formats: Vec<ResponseFormat>,
        /// Model id from the registry.
        #[arg(long, short = 'm', env = "OPENASR_MODEL")]
        model: Option<String>,
        /// Transcription backend: mock or native.
        #[arg(long, value_parser = parse_backend_kind, hide = true)]
        backend: Option<BackendKind>,
        /// Path to an existing ffmpeg binary for preparing recognized non-WAV inputs with the native backend.
        #[arg(long)]
        ffmpeg_bin: Option<PathBuf>,
        /// Label segments with anonymous speakers (SPEAKER_00, SPEAKER_01, ...).
        /// May install the required speaker-diarization capability pack if missing.
        #[arg(long)]
        diarize: bool,
        /// Force an exact speaker count during diarization clustering.
        #[arg(long, requires = "diarize", value_parser = clap::value_parser!(u8).range(1..))]
        speakers: Option<u8>,
        /// Request per-word timestamps from the model's own alignment
        /// (rendered in json/verbose_json and word-timed VTT output).
        #[arg(long)]
        word_timestamps: bool,
        /// Local `.oasr` runtime pack file for native backend transcription.
        #[arg(long)]
        model_pack: Option<PathBuf>,
        /// Local `.oadp` adapter pack (unsigned, base-bound). Fails closed when
        /// it does not match the executing base pack exactly. Phase 0:
        /// moonshine family only.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// Write output to a file (single input) or a directory (multiple
        /// inputs / a directory input). Defaults to stdout for a single input.
        #[arg(long, short = 'o')]
        output: Option<PathBuf>,
        /// With multiple inputs, keep going on per-file errors and report them
        /// at the end instead of stopping at the first failure.
        #[arg(long)]
        continue_on_error: bool,
        /// Print run timing (elapsed, audio duration, real-time factor) instead
        /// of the transcript. Single input only.
        #[arg(long)]
        benchmark: bool,
        /// Download a missing model without the interactive confirmation
        /// (also set by OPENASR_ASSUME_YES).
        #[arg(long, short = 'y')]
        yes: bool,
        /// Never download: fail closed if the resolved model is not installed
        /// (also set by OPENASR_OFFLINE).
        #[arg(long, visible_alias = "no-pull")]
        offline: bool,
        #[command(flatten)]
        longform: NativeLongFormCliOptions,
        #[command(flatten)]
        phrase_bias: PhraseBiasCliOptions,
        #[command(flatten)]
        language_task: LanguageTaskCliOptions,
    },
    /// Manage local voice-match profiles for diarization display names. This is
    /// not authentication; only embeddings are stored.
    Speaker {
        #[command(subcommand)]
        command: SpeakerCommand,
    },
    /// Run the committed performance suite (RTF + peak RSS + WER) and gate
    /// against a baseline.
    BenchSuite {
        /// Committed suite config (TOML).
        #[arg(long, default_value = "perf/suite.toml")]
        config: PathBuf,
        /// Baseline JSON to gate against. Defaults to the suite's sibling
        /// `perf/baselines/` when omitted at the call site.
        #[arg(long)]
        baseline: Option<PathBuf>,
        /// Write measured metrics as a new baseline instead of gating.
        #[arg(long)]
        write_baseline: Option<PathBuf>,
        /// Output format: text, json, markdown.
        #[arg(long, default_value = "markdown", value_parser = parse_benchmark_format)]
        format: BenchmarkFormat,
        /// Only run entries for this family.
        #[arg(long)]
        family: Option<String>,
        /// Runs per entry; the fastest wall-clock sample is kept (best-of-N).
        #[arg(long, default_value_t = 3)]
        runs: usize,
        /// Path to an existing ffmpeg binary for non-WAV audio preparation.
        #[arg(long)]
        ffmpeg_bin: Option<PathBuf>,
        /// Internal: run ONLY this entry id, in-process, and emit its metrics as
        /// a JSON envelope on stdout. The parent spawns one such child per entry
        /// so each entry's peak RSS (a process high-water mark) is uncontaminated
        /// by earlier entries. Not for direct use.
        #[arg(long, hide = true)]
        run_single_entry: Option<String>,
    },
    /// Capture microphone or system audio and print final-per-utterance live captions.
    Live {
        /// Audio source: mic for the default input device, system for loopback/system audio.
        #[arg(long, value_parser = live::parse_live_source, default_value = "mic")]
        source: live::LiveSource,
        /// List available input devices/configs and exit.
        #[arg(long)]
        list_devices: bool,
        /// Optional exact or best-effort microphone device name.
        #[arg(long)]
        device: Option<String>,
        /// Simulate live streaming from a local audio file (WAV/MP3/MP4/M4A/WEBM/FLAC/OGG).
        ///
        /// When set, OpenASR feeds fixed-duration frames from this file into the live pipeline
        /// instead of capturing from microphone.
        #[arg(long)]
        input_file: Option<PathBuf>,
        /// Model id from the registry.
        #[arg(long, short = 'm', env = "OPENASR_MODEL")]
        model: Option<String>,
        /// Transcription backend: mock or native.
        #[arg(long, value_parser = parse_backend_kind, hide = true)]
        backend: Option<BackendKind>,
        /// Local `.oasr` runtime pack file for native backend live transcription.
        #[arg(long)]
        model_pack: Option<PathBuf>,
        /// Output format: text or jsonl.
        #[arg(long, default_value = "text", value_parser = live::parse_live_output_format)]
        format: live::LiveOutputFormat,
        /// Stop after this many seconds.
        #[arg(long)]
        max_seconds: Option<u64>,
        /// Stop after this many completed utterances.
        #[arg(long)]
        max_utterances: Option<usize>,
        /// Realtime frame duration in milliseconds: 10, 20, or 30.
        #[arg(long, default_value_t = 20)]
        frame_duration_ms: u32,
        /// Required speech duration before VAD starts an utterance.
        #[arg(long)]
        speech_start_ms: Option<u32>,
        /// Required silence duration before VAD closes an utterance.
        #[arg(long)]
        speech_stop_ms: Option<u32>,
        /// Audio kept before VAD speech start.
        #[arg(long)]
        pre_roll_ms: Option<u32>,
        /// Maximum utterance duration before forced close.
        #[arg(long)]
        max_utterance_ms: Option<u32>,
        /// Initial no-speech timeout.
        #[arg(long)]
        no_speech_timeout_ms: Option<u32>,
        /// Energy threshold for the MVP VAD.
        #[arg(long)]
        energy_threshold: Option<f32>,
        /// Minimum interval between partial snapshot emissions.
        #[arg(long)]
        partial_interval_ms: Option<u64>,
        /// Sliding-window duration for partial snapshot audio.
        #[arg(long)]
        partial_window_ms: Option<u32>,
        /// Label finalized utterances with anonymous speakers (SPEAKER_00,
        /// SPEAKER_01, ...). May install the required speaker-diarization capability pack.
        #[arg(long)]
        diarize: bool,
        /// Save finalized live transcript history at session end.
        ///
        /// Extension controls export format: .txt, .json, .md, .srt, or .vtt.
        #[arg(long)]
        save: Option<PathBuf>,
        /// Join finalized caption segments into one paragraph when exporting with --save.
        #[arg(long)]
        save_join_segments: bool,
        /// Suggest a conservative title from transcript text when exporting with --save.
        #[arg(long)]
        save_suggest_title: bool,
        /// Update this local text file for OBS Text Source "Read from file" prototype.
        #[arg(long)]
        obs_text_file: Option<PathBuf>,
        /// Max finalized/revised lines to keep in OBS text file updates.
        #[arg(long)]
        obs_max_lines: Option<usize>,
        /// Clear OBS text file on live session start.
        #[arg(long)]
        obs_clear_on_start: bool,
        /// Clear OBS text file on live session stop.
        #[arg(long)]
        obs_clear_on_stop: bool,
        /// Write a local Markdown live session note prototype on stop.
        #[arg(long)]
        markdown_note: Option<PathBuf>,
        /// Append Markdown session note content instead of replacing the file.
        #[arg(long)]
        markdown_append: bool,
        /// Override Markdown session note title.
        #[arg(long)]
        markdown_title: Option<String>,
        /// Suggest a conservative Markdown note title from transcript text.
        #[arg(long)]
        markdown_suggest_title: bool,
        /// Accepted for consistency; live temp WAV utterances normally do not require ffmpeg.
        #[arg(long)]
        ffmpeg_bin: Option<PathBuf>,
        /// Download a missing model without the interactive confirmation
        /// (also set by OPENASR_ASSUME_YES).
        #[arg(long, short = 'y')]
        yes: bool,
        /// Never download: fail closed if the resolved model is not installed
        /// (also set by OPENASR_OFFLINE).
        #[arg(long, visible_alias = "no-pull")]
        offline: bool,
    },
    /// Start the OpenAI-compatible API server.
    ///
    /// Defaults to local HTTP on 127.0.0.1. Non-loopback remote serving must
    /// use HTTPS/WSS and pairing auth.
    Serve {
        /// Address to bind.
        #[arg(long, default_value = "127.0.0.1:8080", env = "OPENASR_ADDR")]
        addr: SocketAddr,
        /// Serve HTTPS/WSS with a generated self-signed certificate; required for non-loopback remote serving.
        #[arg(long)]
        tls_self_signed: bool,
        /// Subject alternative name for the generated self-signed certificate.
        #[arg(long = "tls-san")]
        tls_sans: Vec<String>,
        /// Environment variable containing the pairing administrator token for remote device approval.
        #[arg(long)]
        pairing_admin_token_env: Option<String>,
        /// Model id from the registry.
        #[arg(long, env = "OPENASR_MODEL")]
        model: Option<String>,
        /// Server transcription backend: mock or native.
        #[arg(long, value_parser = parse_backend_kind, hide = true)]
        backend: Option<BackendKind>,
        /// Path to an existing ffmpeg binary for preparing recognized non-WAV uploads.
        #[arg(long)]
        ffmpeg_bin: Option<PathBuf>,
        /// Local `.oasr` runtime pack file for native backend transcription.
        #[arg(long)]
        model_pack: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum ConfigCommand {
    /// Print saved OpenASR config.
    List,
    /// Print one saved config value.
    Get { key: String },
    /// Save one config value.
    Set { key: String, value: String },
    /// Remove one saved config value.
    Unset { key: String },
}

#[derive(Debug, Subcommand)]
pub(crate) enum SpeakerCommand {
    /// Enroll a voice-match profile from a recording.
    Enroll {
        /// 16 kHz mono PCM16 WAV with at least five seconds of speech.
        input: PathBuf,
        /// Display name to use when this voice match wins.
        #[arg(long, default_value = openasr_core::diarize::enrollment::DEFAULT_ENROLLED_NAME)]
        name: String,
        /// Cosine similarity (0-1) required for this voice match.
        #[arg(long)]
        match_similarity: Option<f32>,
    },
    /// Remove all local voice-match profiles.
    Clear,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ModelPackCommand {
    /// Build a local runtime pack (`.oasr`) from model source weights.
    Import {
        #[command(subcommand)]
        command: ImportCommand,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum ImportCommand {
    /// Whisper HF-style source directory into one runtime pack file (`.oasr`).
    #[command(name = "whisper")]
    Whisper {
        /// Source directory containing config.json, tokenizer.json, and model.safetensors.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Package id written to manifest.package.id.
        #[arg(long)]
        package_id: String,
        /// Optional package variant written to manifest.package.variant.
        #[arg(long)]
        package_variant: Option<String>,
        /// Model language written to manifest.model.language.
        #[arg(long, default_value = "en")]
        model_language: String,
        /// Source name written to provenance.source_name.
        #[arg(long, default_value = "openai/whisper")]
        source_name: String,
        /// Source revision written to provenance.source_revision.
        #[arg(long)]
        source_revision: String,
        /// License name written to manifest.license.name.
        #[arg(long, default_value = "MIT")]
        license_name: String,
        /// License source URL/path written to manifest.license.source.
        #[arg(
            long,
            default_value = "https://github.com/openai/whisper/blob/main/LICENSE"
        )]
        license_source: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output.
        #[arg(long, value_enum, default_value_t = ImportWhisperQuantization::Fp16)]
        quantization: ImportWhisperQuantization,
    },
    /// Import one local Qwen ASR HF-style source directory into one runtime pack file (`.oasr`).
    #[command(name = "qwen")]
    Qwen {
        /// Source directory containing config.json, tokenizer artifacts, and one or more *.safetensors files.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Package id written to manifest.package.id.
        #[arg(long)]
        package_id: String,
        /// Optional package variant written to manifest.package.variant.
        #[arg(long)]
        package_variant: Option<String>,
        /// Source name written to provenance.source_name.
        #[arg(long, default_value = "Qwen/Qwen3-ASR")]
        source_name: String,
        /// Source revision written to provenance.source_revision.
        #[arg(long)]
        source_revision: String,
        /// License name written to manifest.license.name.
        #[arg(long, default_value = "Apache-2.0")]
        license_name: String,
        /// License source URL/path written to manifest.license.source.
        #[arg(long)]
        license_source: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output.
        #[arg(long, value_enum, default_value_t = ImportQwen3AsrQuantization::Fp16)]
        quantization: ImportQwen3AsrQuantization,
    },
    /// Import one local Cohere Transcribe HF-style source directory into one runtime pack file (`.oasr`).
    #[command(name = "cohere")]
    Cohere {
        /// Source directory containing config.json, tokenizer.json, and model.safetensors.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Package id written to manifest.package.id.
        #[arg(long)]
        package_id: String,
        /// Optional package variant written to manifest.package.variant.
        #[arg(long)]
        package_variant: Option<String>,
        /// Source name written to provenance.source_name.
        #[arg(long, default_value = "CohereLabs/cohere-transcribe-03-2026")]
        source_name: String,
        /// Source revision written to provenance.source_revision.
        #[arg(long)]
        source_revision: String,
        /// License name written to manifest.license.name.
        #[arg(long, default_value = "Cohere Community License")]
        license_name: String,
        /// License source URL/path written to manifest.license.source.
        #[arg(long)]
        license_source: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output.
        #[arg(long, value_enum, default_value_t = ImportCohereQuantization::Fp16)]
        quantization: ImportCohereQuantization,
    },
    /// Import one local Parakeet-CTC (NVIDIA FastConformer-CTC) HF-style source directory into one runtime pack file (`.oasr`).
    #[command(name = "parakeet-ctc")]
    ParakeetCtc {
        /// Source directory containing config.json, tokenizer.json, and model.safetensors.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Model id written to pack metadata (openasr.model.id).
        #[arg(long)]
        package_id: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output (depthwise convs always stay f16).
        #[arg(long, value_enum, default_value_t = ImportParakeetQuantization::Fp16)]
        quantization: ImportParakeetQuantization,
    },
    /// Import one local Dolphin (WeNet E-Branchformer CTC + attention) source directory into one runtime pack file (`.oasr`).
    #[command(name = "dolphin")]
    Dolphin {
        /// Source directory containing full.safetensors (exported state dict, global_cmvn folded in) and units.txt.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Model id written to pack metadata (openasr.model.id).
        #[arg(long)]
        package_id: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output (context_module/CMVN/mel filterbank always stay f32).
        #[arg(long, value_enum, default_value_t = ImportDolphinQuantization::Fp16)]
        quantization: ImportDolphinQuantization,
    },
    /// Import one local SenseVoiceSmall (FunASR SAN-M/CTC) source directory into one runtime pack file (`.oasr`).
    #[command(name = "sensevoice")]
    Sensevoice {
        /// Source directory containing model.safetensors (from pt_to_safetensors.py), am.mvn, config.yaml, and the SentencePiece bpe model.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Model id written to pack metadata (openasr.model.id).
        #[arg(long)]
        package_id: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output (FSMN kernels/norms always stay f32).
        #[arg(long, value_enum, default_value_t = ImportSensevoiceQuantization::Fp16)]
        quantization: ImportSensevoiceQuantization,
    },
    /// Import one local X-ASR Zipformer2 transducer source directory into one runtime pack file (`.oasr`).
    #[command(name = "xasr-zipformer")]
    XasrZipformer {
        /// Source directory containing config.json, tokens.txt, and model.safetensors.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Model id written to pack metadata (openasr.model.id).
        #[arg(long)]
        package_id: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output.
        #[arg(long, value_enum, default_value_t = ImportXasrZipformerQuantization::Fp16)]
        quantization: ImportXasrZipformerQuantization,
    },
    /// Repackage the pinned upstream Hy-MT2 Q4_K_M GGUF into one translation runtime pack file (`.oasr`).
    ///
    /// Tensor data is preserved byte-for-byte; only `openasr.*` provenance,
    /// translation contract, and license/notice metadata are spliced into the
    /// GGUF KV section. The source file must match the pinned upstream sha256.
    #[command(name = "hymt2-gguf")]
    Hymt2Gguf {
        /// Pinned upstream Hy-MT2-1.8B-Q4_K_M.gguf source file.
        source_gguf: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_pack: PathBuf,
        /// Model id written to pack metadata (openasr.model.id).
        #[arg(long, default_value = "hymt2-1.8b")]
        package_id: String,
        /// Upstream LICENSE.txt file embedded into the pack.
        #[arg(long)]
        license_file: PathBuf,
        /// OpenASR NOTICE.openasr.txt file embedded into the pack; must record the pinned upstream revisions.
        #[arg(long)]
        notice_file: PathBuf,
    },
    /// Import one local wav2vec2-CTC (facebook/wav2vec2-*) HF-style source directory into one runtime pack file (`.oasr`).
    #[command(name = "wav2vec2-ctc")]
    Wav2Vec2Ctc {
        /// Source directory containing config.json, vocab.json, and model.safetensors.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Model id written to pack metadata (openasr.model.id).
        #[arg(long)]
        package_id: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output (conv kernels always stay f16).
        #[arg(long, value_enum, default_value_t = ImportWav2Vec2Quantization::Q4_K)]
        quantization: ImportWav2Vec2Quantization,
    },
    /// Import one local UsefulSensors Moonshine HF-style source directory into one runtime pack file (`.oasr`).
    #[command(name = "moonshine")]
    Moonshine {
        /// Source directory containing config.json, tokenizer.json, and model.safetensors.
        source_root: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Package id written to manifest.package.id.
        #[arg(long)]
        package_id: String,
        /// Optional package variant written to manifest.package.variant.
        #[arg(long)]
        package_variant: Option<String>,
        /// Source name written to provenance.source_name.
        #[arg(long, default_value = "UsefulSensors/moonshine-tiny")]
        source_name: String,
        /// Source revision written to provenance.source_revision.
        #[arg(long, default_value = "main")]
        source_revision: String,
        /// License name written to manifest.license.name.
        #[arg(long, default_value = "MIT")]
        license_name: String,
        /// License source URL/path written to manifest.license.source.
        #[arg(
            long,
            default_value = "https://huggingface.co/UsefulSensors/moonshine-tiny"
        )]
        license_source: String,
        /// Runtime tensor quantization for GGUF-backed `.oasr` output.
        #[arg(long, value_enum, default_value_t = ImportMoonshineQuantization::Fp16)]
        quantization: ImportMoonshineQuantization,
    },
    /// Import a local WeSpeaker ResNet34 speaker-embedder safetensors into one diarization runtime pack (`.oasr`).
    #[command(name = "wespeaker")]
    Wespeaker {
        /// Source WeSpeaker ResNet34 safetensors weight file.
        source_safetensors: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Model id written to pack metadata (openasr.model.id).
        #[arg(long)]
        package_id: String,
        /// Source name written to openasr.source.name.
        #[arg(long, default_value = "pyannote/wespeaker-voxceleb-resnet34-LM")]
        source_name: String,
        /// Source revision written to openasr.source.revision.
        #[arg(long, default_value = "837717ddb9ff5507820346191109dc79c958d614")]
        source_revision: String,
        /// License name written to openasr.license.name. The pyannote/WeSpeaker
        /// VoxCeleb weights are CC-BY-4.0.
        #[arg(long, default_value = "CC-BY-4.0")]
        license_name: String,
        /// License/source URL written to openasr.license.source.
        #[arg(
            long,
            default_value = "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM"
        )]
        license_source: String,
        /// Runtime tensor layout for learned conv/linear kernels. WeSpeaker is f32-only.
        #[arg(long, value_enum, default_value_t = ImportWeSpeakerQuantization::F32)]
        quantization: ImportWeSpeakerQuantization,
    },
    /// Import a local pyannote segmentation-3.0 safetensors into one diarization runtime pack (`.oasr`).
    #[command(name = "pyannote")]
    Pyannote {
        /// Source pyannote-seg safetensors weight file (pyannote_seg.safetensors).
        source_safetensors: PathBuf,
        /// Output path for one runtime pack file (`.oasr`).
        output_root: PathBuf,
        /// Model id written to pack metadata (openasr.model.id).
        #[arg(long)]
        package_id: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum NativeSegmentMode {
    Off,
    Auto,
    Fixed,
    Energy,
    Vad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportWhisperQuantization {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportQwen3AsrQuantization {
    Fp16,
    Q8_0,
    Q3_K,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportCohereQuantization {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportParakeetQuantization {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportSensevoiceQuantization {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportDolphinQuantization {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportXasrZipformerQuantization {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportWav2Vec2Quantization {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[allow(non_camel_case_types)]
pub(crate) enum ImportMoonshineQuantization {
    Fp16,
    Q8_0,
    Q4_K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ImportWeSpeakerQuantization {
    F32,
}
