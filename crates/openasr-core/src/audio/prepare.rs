use std::{fs, path::PathBuf, process::Command};

use crate::{
    BackendKind,
    audio::{
        AudioInputInfo, AudioPreparationError, AudioPreparationOptions, PreparedAudioInput,
        RECOGNIZED_EXTENSIONS,
    },
};

const CONVERSION_STDERR_LIMIT: usize = 800;

pub(crate) fn prepare_external_input(
    info: AudioInputInfo,
    options: &AudioPreparationOptions,
) -> Result<PreparedAudioInput, AudioPreparationError> {
    if options.backend == BackendKind::Native && !options.native_non_wav_requires_conversion {
        let prepared_path = info.path.clone();
        return Ok(PreparedAudioInput {
            original: info,
            prepared_path,
            temp_dir: None,
        });
    }

    if info.extension.as_deref() == Some("wav") {
        let prepared_path = info.path.clone();
        return Ok(PreparedAudioInput {
            original: info,
            prepared_path,
            temp_dir: None,
        });
    }

    if !info.recognized_extension {
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

    let tool = resolve_conversion_tool(options)?;
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
        });
    }

    match fs::metadata(&prepared_path) {
        Ok(metadata) if metadata.is_file() => Ok(PreparedAudioInput {
            original: info,
            prepared_path,
            temp_dir: Some(temp_dir),
        }),
        _ => Err(AudioPreparationError::PreparedFileMissing {
            path: prepared_path,
        }),
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
        hint: missing_converter_hint(),
    })
}

fn missing_converter_hint() -> String {
    #[cfg(target_os = "macos")]
    {
        format!(
            "Install ffmpeg and add it to PATH, pass --ffmpeg-bin /path/to/ffmpeg, set OPENASR_FFMPEG_BIN, run `openasr config set media.ffmpeg_bin /path/to/ffmpeg`, or restore {MACOS_AFCONVERT_PATH} (OpenASR falls back to it automatically when ffmpeg is not configured, but it cannot decode every codec, e.g. Opus/WebM or Ogg Vorbis -- install ffmpeg for full format support)."
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        "Install ffmpeg and add it to PATH, pass --ffmpeg-bin /path/to/ffmpeg, set OPENASR_FFMPEG_BIN, or run `openasr config set media.ffmpeg_bin /path/to/ffmpeg`.".to_string()
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
