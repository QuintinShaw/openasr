use std::{fs, path::PathBuf, process::Command};

use crate::{
    BackendKind,
    audio::{
        AudioInputInfo, AudioPreparationError, AudioPreparationOptions, PreparedAudioInput,
        RECOGNIZED_EXTENSIONS,
    },
};

const FFMPEG_STDERR_LIMIT: usize = 800;

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

    let ffmpeg = resolve_ffmpeg_for_conversion(options)?;
    let temp_dir = tempfile::Builder::new()
        .prefix("openasr-audio-")
        .tempdir()
        .map_err(|source| AudioPreparationError::TempDir { source })?;
    let prepared_path = temp_dir.path().join("prepared.wav");
    let output = Command::new(&ffmpeg)
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-i")
        .arg(&info.path)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg("16000")
        .arg("-c:a")
        .arg("pcm_s16le")
        .arg(&prepared_path)
        .output()
        .map_err(|source| AudioPreparationError::FfmpegSpawn {
            path: ffmpeg.clone(),
            source,
        })?;

    if !output.status.success() {
        return Err(AudioPreparationError::FfmpegFailed {
            backend: options.backend,
            status: output.status.code().map_or_else(
                || "terminated by signal".to_string(),
                |code| code.to_string(),
            ),
            stderr: format_stderr_suffix(&String::from_utf8_lossy(&output.stderr)),
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

fn resolve_ffmpeg_for_conversion(
    options: &AudioPreparationOptions,
) -> Result<PathBuf, AudioPreparationError> {
    let Some(path) = options.ffmpeg_bin.clone() else {
        return Err(AudioPreparationError::MissingFfmpeg {
            backend: options.backend,
        });
    };

    if path.components().count() == 1 {
        return Ok(path);
    }

    match fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() => Ok(path),
        _ => Err(AudioPreparationError::InvalidConfiguredFfmpeg { path }),
    }
}

fn format_stderr_suffix(stderr: &str) -> String {
    let summary = summarize_stderr(stderr);
    if summary.is_empty() {
        String::new()
    } else {
        format!("\nffmpeg stderr: {summary}")
    }
}

fn summarize_stderr(stderr: &str) -> String {
    let summary = stderr.split_whitespace().collect::<Vec<_>>().join(" ");
    if summary.chars().count() <= FFMPEG_STDERR_LIMIT {
        summary
    } else {
        format!(
            "{}...",
            summary
                .chars()
                .take(FFMPEG_STDERR_LIMIT)
                .collect::<String>()
        )
    }
}
