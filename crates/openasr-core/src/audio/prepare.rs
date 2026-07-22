use std::{fs, path::PathBuf, process::Command};

use crate::{
    BackendKind,
    audio::{
        AudioInputInfo, AudioPreparationError, AudioPreparationOptions, PreparedAudioInput,
        RECOGNIZED_EXTENSIONS, decode, symphonia_decode, types::PreparedAudioSamples,
    },
};

const CONVERSION_STDERR_LIMIT: usize = 800;

/// A pass-through `PreparedAudioInput` that hands back `info`'s own path
/// unmodified (the WAV-passthrough branches below): no conversion, no temp
/// dir, no in-memory samples.
fn passthrough(info: AudioInputInfo) -> PreparedAudioInput {
    let prepared_path = info.path.clone();
    PreparedAudioInput {
        original: info,
        samples: PreparedAudioSamples::Path(prepared_path),
        temp_dir: None,
    }
}

pub(crate) fn prepare_external_input(
    info: AudioInputInfo,
    options: &AudioPreparationOptions,
) -> Result<PreparedAudioInput, AudioPreparationError> {
    if options.backend == BackendKind::Native && !options.native_non_wav_requires_conversion {
        return Ok(passthrough(info));
    }

    let is_wav = info.extension.as_deref() == Some("wav");
    if is_wav && wav_is_already_conformant(&info.path) {
        // Already matches the 16 kHz mono PCM16/float32 shape the rest of the
        // pipeline expects: pass it through untouched (cheap, and preserves
        // today's behavior for already-conformant recordings).
        return Ok(passthrough(info));
    }
    if is_wav && !options.ffmpeg_bin_explicit {
        // Non-conformant (other sample rate, stereo, ...) and no explicit
        // ffmpeg was requested: decode via the same in-process symphonia path
        // as the other formats below.
        if let SymphoniaAttempt::Prepared(prepared) = try_symphonia_prepare(&info)? {
            return Ok(prepared);
        }
        // Symphonia could not parse this as a wav at all (corrupt/foreign
        // bytes with a `.wav` extension, or -- per the trust-boundary
        // invariant in AGENTS.md -- a third-party demuxer panic on malformed
        // bytes that `try_symphonia_prepare` already caught and downgraded):
        // preserve today's leniency and pass the original bytes through
        // untouched rather than hard-failing here -- downstream rejects it
        // with a precise WAV-format error if it truly isn't valid input.
        return Ok(passthrough(info));
    }
    // A non-conformant wav with an *explicit* ffmpeg configured falls through
    // to the general external-tool conversion below instead of returning here,
    // so the user's stated intent is actually honored for wav too.

    if !is_wav && !info.recognized_extension {
        let description = info
            .extension
            .as_deref()
            .map(|extension| format!("extension .{extension} is not recognized"))
            .unwrap_or_else(|| "the file has no extension".to_string());
        return Err(AudioPreparationError::UnsupportedInput {
            backend: options.backend,
            description,
            extensions: RECOGNIZED_EXTENSIONS.join(", "),
        });
    }

    // In-process decode is the default main path for every other recognized
    // format (m4a/AAC-LC/ALAC, mp4, qta, mp3, flac, ogg/vorbis, mkv/webm
    // vorbis). It only ever falls through (never a hard error) when the
    // container/codec is not supported (e.g. HE-AAC, Opus in any container),
    // the file is malformed, or -- a third-party demuxer bug on adversarial
    // input -- the underlying symphonia call panicked (caught and downgraded
    // by `try_symphonia_prepare`, per the panic-free trust-boundary invariant
    // in AGENTS.md); in all three cases control falls through to the external
    // ffmpeg/afconvert chain below exactly as before. An explicitly
    // configured ffmpeg binary is an escape hatch that always wins, so it is
    // checked first.
    let diagnostic = if !options.ffmpeg_bin_explicit {
        match try_symphonia_prepare(&info)? {
            SymphoniaAttempt::Prepared(prepared) => return Ok(prepared),
            SymphoniaAttempt::NotHandled { codec_label } => Diagnostic::from(codec_label),
            SymphoniaAttempt::ParserPanicked => Diagnostic::ParserPanicked,
        }
    } else {
        // The explicit-ffmpeg escape hatch skips the in-process decode
        // attempt entirely, so this is the only symphonia probe on this
        // path -- still worth doing purely for the diagnostic codec name in
        // case the configured ffmpeg also fails to convert the file.
        match symphonia_decode::probe_codec_label(&info.path, info.extension.as_deref()) {
            symphonia_decode::ProbeOutcome::Codec(label) => Diagnostic::Codec(label),
            symphonia_decode::ProbeOutcome::Unknown => Diagnostic::Unknown,
            symphonia_decode::ProbeOutcome::ParserPanicked => Diagnostic::ParserPanicked,
        }
    };

    let tool = resolve_conversion_tool(options, &diagnostic)?;
    let temp_dir = tempfile::Builder::new()
        .prefix("openasr-audio-")
        .tempdir()
        .map_err(|source| AudioPreparationError::TempDir { source })?;
    let prepared_path = temp_dir.path().join("prepared.wav");
    let output = tool
        .build_command(&info.path, &prepared_path)
        .output()
        .map_err(|source| AudioPreparationError::ConversionSpawn {
            tool: tool.label().to_string(),
            path: tool.path().clone(),
            source,
        })?;

    if !output.status.success() {
        return Err(AudioPreparationError::ConversionFailed {
            backend: options.backend,
            tool: tool.label().to_string(),
            status: output.status.code().map_or_else(
                || "terminated by signal".to_string(),
                |code| code.to_string(),
            ),
            stderr: format_stderr_suffix(tool.label(), &String::from_utf8_lossy(&output.stderr)),
            codec_note: codec_note(&diagnostic),
        });
    }

    match fs::metadata(&prepared_path) {
        Ok(metadata) if metadata.is_file() => Ok(PreparedAudioInput {
            original: info,
            samples: PreparedAudioSamples::Path(prepared_path),
            temp_dir: Some(temp_dir),
        }),
        _ => Err(AudioPreparationError::PreparedFileMissing {
            path: prepared_path,
        }),
    }
}

fn wav_is_already_conformant(path: &std::path::Path) -> bool {
    matches!(
        decode::probe_wav_pcm_shape(path),
        Ok(Some(fmt)) if fmt.channels == 1
            && fmt.sample_rate == 16_000
            && matches!((fmt.audio_format, fmt.bits_per_sample), (1, 16) | (3, 32))
    )
}

/// Outcome of [`try_symphonia_prepare`].
enum SymphoniaAttempt {
    /// Decoded straight to memory; ready to use.
    Prepared(PreparedAudioInput),
    /// Not decodable in-process; fall back to the external converter chain.
    /// `codec_label` is the codec name the demuxer identified, if any (see
    /// `symphonia_decode::SymphoniaOutcome::Unsupported`).
    NotHandled { codec_label: Option<String> },
    /// The underlying symphonia demuxer/decoder panicked on this input (a
    /// third-party bug on adversarial bytes, e.g. `symphonia-format-mkv`'s
    /// vint underflow -- see `symphonia_decode`'s module docs). Already
    /// caught there; callers must not treat this as a hard error, only as a
    /// reason to fall back, same as `NotHandled`.
    ParserPanicked,
}

/// Tries the in-process symphonia decode path for `info`. Never a hard error
/// on an unsupported/malformed/panicking input -- the caller falls back to
/// the external converter chain in every such case (see [`SymphoniaAttempt`]).
///
/// On success the decoded samples stay resident in memory
/// (`PreparedAudioSamples::InMemory`) instead of being encoded to a WAV,
/// written to a temp file, and immediately re-read + re-parsed back into the
/// exact same samples by the downstream consumer -- the write-then-reread
/// round trip this used to always pay for every non-WAV (and non-conformant
/// WAV) input.
fn try_symphonia_prepare(info: &AudioInputInfo) -> Result<SymphoniaAttempt, AudioPreparationError> {
    let (samples, source_format) =
        match symphonia_decode::try_decode_to_pcm16_mono_16k(&info.path, info.extension.as_deref())
        {
            symphonia_decode::SymphoniaOutcome::Decoded(samples, source_format) => {
                (samples, source_format)
            }
            symphonia_decode::SymphoniaOutcome::Unsupported { codec_label } => {
                return Ok(SymphoniaAttempt::NotHandled { codec_label });
            }
            symphonia_decode::SymphoniaOutcome::ParserPanicked => {
                return Ok(SymphoniaAttempt::ParserPanicked);
            }
        };

    // The probe stage (`probe::probe_audio_details`) only reads source
    // format off WAV's fmt chunk; for the non-wav formats that land here it
    // could not have known this yet, so fill it in now from the decode that
    // just ran -- the true source format, not a second separate probe.
    let mut original = info.clone();
    original.sample_rate_hz = Some(source_format.sample_rate_hz);
    original.channels = Some(source_format.channels);

    Ok(SymphoniaAttempt::Prepared(PreparedAudioInput {
        original,
        samples: PreparedAudioSamples::InMemory(samples.into()),
        temp_dir: None,
    }))
}

/// What (if anything) is known about why the in-process symphonia path
/// didn't produce a result, for building the error message if the external
/// converter subsequently also fails.
enum Diagnostic {
    /// The demuxer identified the codec (whether or not a decoder for it is
    /// linked in).
    Codec(String),
    /// Nothing more specific than "not handled" is known.
    Unknown,
    /// The symphonia demuxer/decoder itself panicked on this input; see
    /// `symphonia_decode`'s module docs. Distinguished from `Unknown` so the
    /// error can say "internal parser error" instead of implying an
    /// unsupported-but-well-formed codec.
    ParserPanicked,
}

impl From<Option<String>> for Diagnostic {
    fn from(codec_label: Option<String>) -> Self {
        match codec_label {
            Some(label) => Self::Codec(label),
            None => Self::Unknown,
        }
    }
}

/// An external tool used to convert a non-WAV input into a 16 kHz mono PCM16
/// WAV. macOS ships `/usr/bin/afconvert` on every install (no Homebrew
/// required), so it is used as a fallback conversion path when ffmpeg is not
/// configured and cannot be found on PATH.
enum ConversionTool {
    Ffmpeg(PathBuf),
    #[cfg(target_os = "macos")]
    Afconvert(PathBuf),
}

impl ConversionTool {
    fn label(&self) -> &'static str {
        match self {
            Self::Ffmpeg(_) => "ffmpeg",
            #[cfg(target_os = "macos")]
            Self::Afconvert(_) => "afconvert",
        }
    }

    fn path(&self) -> &PathBuf {
        match self {
            Self::Ffmpeg(path) => path,
            #[cfg(target_os = "macos")]
            Self::Afconvert(path) => path,
        }
    }

    fn build_command(&self, input: &std::path::Path, output: &std::path::Path) -> Command {
        match self {
            Self::Ffmpeg(path) => {
                let mut command = Command::new(path);
                command
                    .arg("-hide_banner")
                    .arg("-loglevel")
                    .arg("error")
                    .arg("-y")
                    .arg("-i")
                    .arg(input)
                    .arg("-vn")
                    .arg("-ac")
                    .arg("1")
                    .arg("-ar")
                    .arg("16000")
                    .arg("-c:a")
                    .arg("pcm_s16le")
                    .arg(output);
                command
            }
            #[cfg(target_os = "macos")]
            Self::Afconvert(path) => {
                let mut command = Command::new(path);
                // -f WAVE -d LEI16@16000 -c 1: canonical 16 kHz mono PCM16 WAV,
                // matching the ffmpeg path above (afconvert always writes the
                // fmt chunk as WAVE_FORMAT_EXTENSIBLE; the WAV reader in
                // `api::audio_io` unwraps that to the underlying PCM/float
                // subformat).
                command
                    .arg("-f")
                    .arg("WAVE")
                    .arg("-d")
                    .arg("LEI16@16000")
                    .arg("-c")
                    .arg("1")
                    .arg(input)
                    .arg(output);
                command
            }
        }
    }
}

/// macOS system path for `afconvert`, present on every macOS install
/// (Core Audio command-line tool, no Homebrew/ffmpeg required).
#[cfg(target_os = "macos")]
const MACOS_AFCONVERT_PATH: &str = "/usr/bin/afconvert";

fn resolve_conversion_tool(
    options: &AudioPreparationOptions,
    diagnostic: &Diagnostic,
) -> Result<ConversionTool, AudioPreparationError> {
    if let Some(path) = options.ffmpeg_bin.clone() {
        if path.components().count() == 1 {
            return Ok(ConversionTool::Ffmpeg(path));
        }
        return match fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => Ok(ConversionTool::Ffmpeg(path)),
            _ => Err(AudioPreparationError::InvalidConfiguredFfmpeg { path }),
        };
    }

    #[cfg(target_os = "macos")]
    {
        let afconvert = PathBuf::from(MACOS_AFCONVERT_PATH);
        if matches!(fs::metadata(&afconvert), Ok(metadata) if metadata.is_file()) {
            return Ok(ConversionTool::Afconvert(afconvert));
        }
    }

    Err(AudioPreparationError::MissingFfmpeg {
        backend: options.backend,
        hint: missing_converter_hint(diagnostic),
    })
}

/// A short extra sentence for error messages describing what's known about
/// why the in-process decode didn't handle this file; empty string (no extra
/// sentence) when nothing more specific than "unsupported" is known.
fn codec_note(diagnostic: &Diagnostic) -> String {
    match diagnostic {
        Diagnostic::Codec(label) => format!(
            "\nDetected audio codec: {label}. OpenASR's built-in decoder supports AAC, ALAC, FLAC, MP3, PCM/WAV, and Vorbis, but not {label} in-process."
        ),
        Diagnostic::ParserPanicked => {
            "\nOpenASR's built-in parser hit an internal error while inspecting this file. This looks like a malformed or corrupted container (or an edge case the parser doesn't handle), not merely an unsupported codec.".to_string()
        }
        Diagnostic::Unknown => String::new(),
    }
}

fn missing_converter_hint(diagnostic: &Diagnostic) -> String {
    let codec_phrase = match diagnostic {
        Diagnostic::Codec(label) => format!("this format ({label})"),
        Diagnostic::ParserPanicked => {
            "this file (its container looks malformed or corrupted, or hits an edge case the bundled parser doesn't handle)".to_string()
        }
        Diagnostic::Unknown => {
            "this format (e.g. HE-AAC, Opus, or an unrecognized WebM track)".to_string()
        }
    };
    #[cfg(target_os = "macos")]
    {
        format!(
            "OpenASR's built-in decoder does not support {codec_phrase}; it needs ffmpeg. Install ffmpeg and add it to PATH, pass --ffmpeg-bin /path/to/ffmpeg, set OPENASR_FFMPEG_BIN, run `openasr config set media.ffmpeg_bin /path/to/ffmpeg`, or restore {MACOS_AFCONVERT_PATH} (OpenASR falls back to it automatically when ffmpeg is not configured, but it cannot decode every codec either -- install ffmpeg for full format support)."
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        format!(
            "OpenASR's built-in decoder does not support {codec_phrase}; it needs ffmpeg. Install ffmpeg and add it to PATH, pass --ffmpeg-bin /path/to/ffmpeg, set OPENASR_FFMPEG_BIN, or run `openasr config set media.ffmpeg_bin /path/to/ffmpeg`."
        )
    }
}

fn format_stderr_suffix(tool: &str, stderr: &str) -> String {
    let summary = summarize_stderr(stderr);
    if summary.is_empty() {
        String::new()
    } else {
        format!("\n{tool} stderr: {summary}")
    }
}

fn summarize_stderr(stderr: &str) -> String {
    let summary = stderr.split_whitespace().collect::<Vec<_>>().join(" ");
    if summary.chars().count() <= CONVERSION_STDERR_LIMIT {
        summary
    } else {
        format!(
            "{}...",
            summary
                .chars()
                .take(CONVERSION_STDERR_LIMIT)
                .collect::<String>()
        )
    }
}
