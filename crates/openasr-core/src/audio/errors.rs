use std::path::PathBuf;

use thiserror::Error;

use crate::BackendKind;

#[derive(Debug, Error)]
pub enum AudioInputError {
    #[error("Input file not found: {path}\nPlease provide a valid audio or video file path.")]
    NotFound { path: PathBuf },
    #[error("Input path is a directory: {path}\nPlease provide a valid audio or video file path.")]
    Directory { path: PathBuf },
    #[error(
        "Input path is not a regular file: {path}\nPlease provide a valid audio or video file path."
    )]
    NotRegularFile { path: PathBuf },
    #[error("Could not read input file: {path}\nPlease check the path and file permissions.")]
    Metadata { path: PathBuf },
}

#[derive(Debug, Error)]
pub enum AudioPreparationError {
    #[error("{0}")]
    Input(#[from] AudioInputError),
    #[error(
        "Unsupported audio input for the {backend} backend: {description}\nOpenASR can pass WAV files directly and can prepare recognized non-WAV inputs ({extensions}) through local ffmpeg for non-mock backends. Convert this file to WAV, use a recognized extension, or use the mock backend for plumbing tests."
    )]
    UnsupportedInput {
        backend: BackendKind,
        description: String,
        extensions: String,
    },
    #[error(
        "Input requires audio conversion before the {backend} backend can read it, but ffmpeg was not found.\nInstall ffmpeg and add it to PATH, pass --ffmpeg-bin /path/to/ffmpeg, set OPENASR_FFMPEG_BIN, or run `openasr config set media.ffmpeg_bin /path/to/ffmpeg`."
    )]
    MissingFfmpeg { backend: BackendKind },
    #[error(
        "Configured ffmpeg binary was not found or is not a regular file: {path}\nPass --ffmpeg-bin /path/to/ffmpeg, set OPENASR_FFMPEG_BIN, or run `openasr config set media.ffmpeg_bin /path/to/ffmpeg` with a valid ffmpeg executable."
    )]
    InvalidConfiguredFfmpeg { path: PathBuf },
    #[error(
        "Could not create temporary audio preparation directory.\nPlease check your temporary directory permissions. Details: {source}"
    )]
    TempDir { source: std::io::Error },
    #[error(
        "Could not convert input audio for the {backend} backend with ffmpeg (status {status}).\nOpenASR prepares recognized non-WAV inputs as temporary 16 kHz mono PCM WAV for non-mock backends. Check that your local ffmpeg can decode this container/codec, pass a different --ffmpeg-bin, or convert the file to WAV yourself.{stderr}"
    )]
    FfmpegFailed {
        backend: BackendKind,
        status: String,
        stderr: String,
    },
    #[error(
        "Could not run ffmpeg: {path}\nPlease check that the file exists and is executable, or configure ffmpeg with --ffmpeg-bin, OPENASR_FFMPEG_BIN, or `openasr config set media.ffmpeg_bin`. Details: {source}"
    )]
    FfmpegSpawn {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "ffmpeg completed but did not create a prepared WAV file: {path}\nPlease check that your local ffmpeg can write to the temporary directory."
    )]
    PreparedFileMissing { path: PathBuf },
}
